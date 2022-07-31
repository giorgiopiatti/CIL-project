import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (283228, 193886)

"""
Base model

"""
class NCFDistribution(pl.LightningModule):

    def __init__(self, 
    embedding_size_movie=16, 
    embedding_size_user=16,
    hidden_size = 16,
    lr=1e-3, 
    alpha=0.1,
    sigma_prior=1.0,
    distance_0_to_3 = 0,
    distance_3_to_2 = 0.25,
    distance_2_to_1 = 0.25,
    distance_0_to_4 = 0.25,
    distance_4_to_5 = 0.25,
    scaling= 1.0,
    p_dropout=0.2,
    weight_decay=0,
    loss_use_class_weights=False
    ):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.loss_use_class_weights = loss_use_class_weights
        self.weight_decay = weight_decay

        self.sigma_prior=sigma_prior
        self.distance_0_to_3 = distance_0_to_3
        self.distance_3_to_2 = distance_3_to_2
        self.distance_2_to_1 = distance_2_to_1
        self.distance_0_to_4 = distance_0_to_4
        self.distance_4_to_5 = distance_4_to_5
        self.scaling = scaling
        
        self.create_model(embedding_size_movie, embedding_size_user, hidden_size, p_dropout)
    

    def create_model(self,embedding_size_movie, embedding_size_user, hidden_size, p_dropout):
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size_user)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size_movie)

        
        
        self.prob_dist =  nn.Sequential(
            nn.Linear(embedding_size_user + embedding_size_movie, hidden_size),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    
    def pdf_gaussian(self, x, mu, sigma):
       return torch.div(torch.exp(-0.5* torch.pow((x-mu)/sigma, 2)), sigma*torch.sqrt(torch.tensor(2*torch.pi, device=self.device)))

    def kl_divergence(self, sigma1, sigma2):
        return torch.log(sigma2/sigma1) + (torch.pow(sigma1, 2)) / (2.*torch.pow(sigma2, 2))
     
    def penalty_term(self, mu, sigma):
        sigma_prior = torch.tensor(1.0, device=self.device)
        return self.kl_divergence(sigma1=sigma_prior, sigma2=sigma) 

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        
        prod_dist = self.prob_dist(concat)

        mu = self.scaling*torch.tanh(prod_dist[:, 0])
        sigma = torch.log(1 + torch.exp(prod_dist[:, 1]) + 0.01)
        
        
        rating_1_prob = self.pdf_gaussian(-self.distance_0_to_3-self.distance_3_to_2 -self.distance_2_to_1, mu, sigma) + 0.01
        rating_2_prob = self.pdf_gaussian(-self.distance_0_to_3-self.distance_3_to_2, mu, sigma) + 0.01
        rating_3_prob = self.pdf_gaussian(0.-self.distance_0_to_3, mu, sigma) + 0.01
        rating_4_prob = self.pdf_gaussian(self.distance_0_to_4, mu, sigma) + 0.01
        rating_5_prob = self.pdf_gaussian(self.distance_0_to_4 + self.distance_4_to_5, mu, sigma) + 0.01

        prob_sum =  rating_1_prob + rating_2_prob + rating_3_prob + rating_4_prob + rating_5_prob
        rating_1_prob /= prob_sum
        rating_2_prob /= prob_sum
        rating_3_prob /= prob_sum
        rating_4_prob /= prob_sum
        rating_5_prob /= prob_sum


        probs = torch.hstack([
            rating_1_prob.unsqueeze(-1), 
            rating_2_prob.unsqueeze(-1), 
            rating_3_prob.unsqueeze(-1), 
            rating_4_prob.unsqueeze(-1), 
            rating_5_prob.unsqueeze(-1) 
        ])
    
        probs = torch.log(probs)
    
        return probs, mu, sigma
    
    def loss(self, yhat, y):
        
        weights = torch.tensor([0.03696667, 0.08426852, 0.23308257, 0.27588211, 0.36980013], device=self.device)
        if self.loss_use_class_weights:
            criterion = nn.NLLLoss(weight=weights)
        else:
            criterion = nn.NLLLoss()
        return criterion(yhat, y)

    
    def rmse_metric(self, yhat, y):
        """
        RMSE
        """
        yhat = yhat.type(torch.float)
        y = y.type(torch.float)
        return torch.sqrt(torch.mean((yhat-y)**2))

    def configure_optimizers(self):
        # if self.swa:
        #     return torch.optim.SGD(self.parameters(), self.lr)
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)

        #return swd_optim.AdamS(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)

    
    def get_expected(self, pred):
        return 1.0*torch.exp(pred[:, 0]) +  2.0*torch.exp(pred[:, 1]) +  3.0*torch.exp(pred[:, 2]) +  4.0*torch.exp(pred[:, 3]) +  5.0*torch.exp(pred[:, 4])
    
    def training_step(self, batch, batch_idx):
        x,y = batch

        pred, mu, sigma = self(x)
        #_, yhat = torch.max(pred, 1)
        yhat = self.get_expected(pred) - 1
        penalty = torch.mean(self.penalty_term(mu, sigma))
        nll = self.loss(pred, y)

        loss = nll + self.alpha*penalty

        mse = self.rmse_metric(y, yhat)

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_nll', nll, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_penalty', penalty, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_rmse', mse, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        pred, _, _ = self(x)
        #_, yhat = torch.max(pred, 1)
        yhat = self.get_expected(pred) -1

        nll = self.loss(pred, y)
        mse = self.rmse_metric(y, yhat)
        self.log('val_loss', nll, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_rmse', mse, on_epoch=True, on_step=True, prog_bar=True)

    
    def predict_step(self, batch, batch_idx):
        pred, mu, sigma = self(batch)
        # _, yhat = torch.max(pred, 1)
        # yhat = yhat + 1
        return  self.get_expected(pred)

