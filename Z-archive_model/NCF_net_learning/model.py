from ast import Mod
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)

"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size_user, emb_size_movie, p_dropout, lr, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.user_embedding = nn.Embedding(number_of_users, emb_size_user, scale_grad_by_freq=True)
        self.movie_embedding = nn.Embedding(number_of_movies, emb_size_movie, scale_grad_by_freq=True)
        
        self.bias_user = nn.Embedding(number_of_users, 1)
        self.bias_movie = nn.Embedding(number_of_movies, 1)

        number_of_weights = emb_size_user*emb_size_user + emb_size_user*2 + 2
        self.f_m = nn.Sequential(
            nn.Linear(emb_size_movie, number_of_weights*2),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
            # nn.Linear(number_of_weights*2, number_of_weights*2),
            # nn.LeakyReLU(),
            # nn.Dropout(p_dropout),
            nn.Linear(number_of_weights*2, number_of_weights)
        )

        self.emb_size_user = emb_size_user
        self.lr = lr
        self.weight_decay = weight_decay
        

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
        layer_3_weights = weights_f_m[:, -2:].unsqueeze(-1)

        # print('layer_1_weights', layer_1_weights.shape)
        # print('layer_2_weights', layer_2_weights.shape)
        # print('layer_3_weights', layer_3_weights.shape)
        
        out = torch.bmm(users_embedding.unsqueeze(1),layer_1_weights)
        # print('out1', out.shape)
        out = torch.nn.functional.leaky_relu(out)
        out = torch.bmm(out, layer_2_weights)
        # print('out2', out.shape)
        out = torch.nn.functional.leaky_relu(out)

        out = torch.bmm(out,layer_3_weights).squeeze(1)
        # print('out23', out.shape)
       
        #return torch.nn.functional.relu(out)
        #return 2*torch.nn.functional.tanh(out + self.bias_movie(movies) + self.bias_user(users)) + 3
        return out + self.bias_movie(movies) + self.bias_user(users)
    
    def loss(self, yhat, y):
        """
        RMSE
        """
        return torch.sqrt(torch.mean((yhat -y)**2))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        self.log('train_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        self.log('val_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
