from ast import Mod
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size_user, emb_size_movie, p_dropout, lr=1e-3, 
        alpha=0.1,
        sigma_prior=1.0,
        distance_0_to_3 = 0,
        distance_3_to_2 = 0.25,
        distance_2_to_1 = 0.25,
        distance_0_to_4 = 0.25,
        distance_4_to_5 = 0.25,
        scaling= 1.0,
        weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(number_of_users, emb_size_user)
        self.movie_embedding = nn.Embedding(number_of_movies, emb_size_movie)
        
        self.bias_user = nn.Embedding(number_of_users, 1)
        self.bias_movie = nn.Embedding(number_of_movies, 1)

        number_of_weights = emb_size_user*emb_size_user + emb_size_user*2
        self.f_m = nn.Sequential(
            nn.Linear(emb_size_movie, number_of_weights*2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            # nn.Linear(number_of_weights*2, number_of_weights*2),
            # nn.SELU(),
            # nn.Dropout(p_dropout),
            nn.Linear(number_of_weights*2, number_of_weights)
        )


        self.emb_size_user = emb_size_user
        self.lr = lr
        self.weight_decay = weight_decay

        self.sigma_prior=sigma_prior
        self.distance_0_to_3 = distance_0_to_3
        self.distance_3_to_2 = distance_3_to_2
        self.distance_2_to_1 = distance_2_to_1
        self.distance_0_to_4 = distance_0_to_4
        self.distance_4_to_5 = distance_4_to_5
        self.scaling = scaling
        self.alpha = alpha
        

    def forward(self, batch):
        users, movies = batch[:, 0], batch[:, 1]

        users_embedding = self.user_embedding(users)
        movies_embedding = self.movie_embedding(movies)
        # print('users_embedding', users_embedding.shape)
        # print('movies_embedding', movies_embedding.shape)


        weights_f_m = self.f_m(movies_embedding) # ev use movies [0] assuming forward is called using of fix movie at each time
        #batch_size, number_of_weights
        # print('weights_f_m', weights_f_m.shape)

        batch_size = weights_f_m.shape[0]
        layer_1_weights = weights_f_m[:, 0:self.emb_size_user*self.emb_size_user].reshape(batch_size, self.emb_size_user, self.emb_size_user)
        layer_2_weights = weights_f_m[:, self.emb_size_user*self.emb_size_user: self.emb_size_user*self.emb_size_user +  self.emb_size_user*2].reshape(batch_size, self.emb_size_user,2 )
      
        # print('layer_1_weights', layer_1_weights.shape)
        # print('layer_2_weights', layer_2_weights.shape)
  
        out = torch.bmm(users_embedding.unsqueeze(1),layer_1_weights)
        # print('out1', out.shape)
        out = torch.nn.functional.leaky_relu(out)
        out = torch.bmm(out, layer_2_weights).squeeze(1)
        # print('out2', out.shape)
       
        #DIST code ---------------------------------
        mu = self.scaling*torch.tanh(out[:, 0])
        sigma = torch.log(1 + torch.exp(out[:, 1]) + 0.01)
        
     
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
    
    def pdf_gaussian(self, x, mu, sigma):
       return torch.div(torch.exp(-0.5* torch.pow((x-mu)/sigma, 2)), sigma*torch.sqrt(torch.tensor(2*torch.pi, device=self.device)))

    def kl_divergence(self, sigma1, sigma2):
        return torch.log(sigma2/sigma1) + (torch.pow(sigma1, 2)) / (2.*torch.pow(sigma2, 2))
     
    def penalty_term(self, mu, sigma):
        sigma_prior = torch.tensor(1.0, device=self.device)
        return self.kl_divergence(sigma1=sigma_prior, sigma2=sigma) 


    def loss(self, yhat, y):
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
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
    
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