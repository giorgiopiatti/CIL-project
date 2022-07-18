from turtle import forward
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)



class SmallNCF(nn.Module):

    def __init__(self, embedding_size, hidden_size, p_dropout):
        super().__init__()
        self.prob_dist =  nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, batch):
        prod_dist = self.prob_dist(batch)

        mu = torch.tanh(prod_dist[:, 0])
        sigma = torch.log(1 + torch.exp(prod_dist[:, 1]) + 0.01)
        return mu, sigma


"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, 
  
    number_of_factors=4,
    factor_embedding_size=4,
    last_factor_size = 8,

    factor_hidden_size = 16,
    last_factor_hidden_size = 16,
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
        
        self.number_of_factors = number_of_factors
        self.factor_embedding_size = factor_embedding_size

        self.embedding_layer_users = nn.Embedding(number_of_users, number_of_factors*factor_embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, number_of_factors*factor_embedding_size)

        self.embedding_layer_users_last = nn.Embedding(number_of_users, last_factor_size)
        self.embedding_layer_movies_last = nn.Embedding(number_of_movies,last_factor_size)

        self.ncf_factors = nn.ModuleList()
        for _ in range(number_of_factors):
            self.ncf_factors.append(SmallNCF(2*factor_embedding_size, factor_hidden_size, p_dropout)) 
       
        self.ncf_last_factor = SmallNCF(last_factor_size*2, last_factor_hidden_size, p_dropout)

        self.sigma_coeffs = torch.nn.parameter.Parameter(torch.ones(size=(number_of_factors+1,), device=self.device))
        self.mu_coeffs = torch.nn.parameter.Parameter(torch.ones(size=(number_of_factors+1,), device=self.device))

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
        
        users_embedding = users_embedding.reshape(users_embedding.shape[0], self.number_of_factors, self.factor_embedding_size).permute([1,0,2])
        movies_embedding = movies_embedding.reshape(movies_embedding.shape[0], self.number_of_factors, self.factor_embedding_size).permute([1,0,2])
        
        last_users_embedding = self.embedding_layer_users_last(users)
        last_movies_embedding = self.embedding_layer_movies_last(movies)
        
        mus = []
        sigmas = []
        for i in range(self.number_of_factors):
            concat = torch.cat([users_embedding[i], movies_embedding[i]], dim=1)
            ncf_mu, ncf_sigma = self.ncf_factors[i](concat)
            mus.append(ncf_mu)
            sigmas.append(ncf_sigma)
        
        concat = torch.cat([last_users_embedding, last_movies_embedding], dim=1)
        ncf_mu, ncf_sigma = self.ncf_last_factor(concat)
      
        mus.append(ncf_mu)
        sigmas.append(ncf_sigma)

        all_mu = torch.stack(mus, dim=1)
        all_sigma = torch.stack(sigmas, dim=1)

        mu = self.scaling*torch.matmul(all_mu, torch.nn.functional.softmax(self.mu_coeffs))
        sigma = torch.matmul(all_sigma,  torch.nn.functional.softmax(self.sigma_coeffs))
        
        
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


