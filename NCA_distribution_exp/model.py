from turtle import forward
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup


class MultiHeadAttention(nn.Module):
    def __init__(self, in_size, out_size, emb_size, num_heads, 
        hidden_size_1, 
        hidden_size_2, device) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        
        self.q = nn.Sequential(
             nn.Linear(in_size, hidden_size_1),
             nn.ReLU(),
             nn.Linear(hidden_size_1, hidden_size_2),
             nn.ReLU(),
             nn.Linear(hidden_size_2, emb_size)
        )

        self.v = nn.Sequential(
             nn.Linear(in_size, hidden_size_1),
             nn.ReLU(),
             nn.Linear(hidden_size_1, hidden_size_2),
             nn.ReLU(),
             nn.Linear(hidden_size_2, emb_size)
        )

        self.k = nn.Sequential(
             nn.Linear(in_size, hidden_size_1),
             nn.ReLU(),
             nn.Linear(hidden_size_1, hidden_size_2),
             nn.ReLU(),
             nn.Linear(hidden_size_2, emb_size)
        )
        
        # self.q = nn.Linear(in_size, emb_size, bias=False)
        # self.k = nn.Linear(in_size, emb_size, bias=False)
        # self.v = nn.Linear(in_size, emb_size)

        self.Wo = nn.Linear(emb_size,out_size)
        self.device = device



    def split_heads(self, x):
        #x.shape = (number_of_users, emb_size)
        depth = self.emb_size // self.num_heads
        x = torch.reshape(x, (number_of_users, self.num_heads, depth))
        #x.shape = (number_of_users, num_heads, depth)
        return torch.permute(x, [1,0,2])   #(num_heads, number_of_users, depth)



    def scaled_relu_attention(self, q, k, v, mask):
        # q,k,v (..., num_heads, attn_dim, emb_size)
        q = torch.relu(q)
        k = torch.relu(k)

        matmul_qk = torch.matmul(q, k.transpose(1,2) )

        # scale matmul_qk
        dk = torch.tensor(k.shape[-1], device=self.device, dtype=torch.float)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # normalized on the last axis (seq_len_k) so that the scores add up to 1.
        #attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
     
        attention_scores_sum = torch.sum(scaled_attention_logits, dim=-1).unsqueeze(-1) + 1e-8 #prevent division by zero


        #print('scaled_attention_logits', scaled_attention_logits.shape)
        #print('attention_scores_sum', attention_scores_sum.shape)
        #print(scaled_attention_logits[0][0].sum())
        #print(attention_scores_sum[0])
        attention_weights  = torch.div(scaled_attention_logits, attention_scores_sum)
        
        #print(attention_weights[0][0].sum())
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, batch):
        batch_shape = batch.shape #(number_of_users, in_size)

        v = self.v(batch) 
        q = self.q(batch)
        k = self.k(batch)

        v = self.split_heads(v)
        q = self.split_heads(q)
        k = self.split_heads(k)
        # q,k,v (num_heads, number_of_users, depth)


        att_out, _ = self.scaled_relu_attention(q, k, v, None)
        # (num_heads, number_of_users, emb_size//num_heads)
        # concatenate the outputs from different heads
        att_out = att_out.permute((1,0,2))    # (number_of_users, num_heads, emb_size//num_heads)
        att_out = att_out.reshape((number_of_users, self.emb_size)) #(number_of_users, emb_size)

        out = self.Wo(att_out)
        return out
        
"""
Base model

"""
class Model(pl.LightningModule):

    def __init__(self, emb_size,  
        hidden_size_1, 
        hidden_size_2, num_heads, lr, weight_decay, 
        
        alpha=0.1,
        sigma_prior=1.0,
        distance_0_to_3 = 0,
        distance_3_to_2 = 0.25,
        distance_2_to_1 = 0.25,
        distance_0_to_4 = 0.25,
        distance_4_to_5 = 0.25,
        scaling= 1.0,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
    
        self.mha = MultiHeadAttention(number_of_movies, 2*number_of_movies, emb_size*num_heads, num_heads, 
                hidden_size_1, 
                hidden_size_2,
                device=self.device)
        self.alpha = alpha
        self.sigma_prior=sigma_prior
        self.distance_0_to_3 = distance_0_to_3
        self.distance_3_to_2 = distance_3_to_2
        self.distance_2_to_1 = distance_2_to_1
        self.distance_0_to_4 = distance_0_to_4
        self.distance_4_to_5 = distance_4_to_5
        self.scaling = scaling

    def forward(self, batch):
        out = self.mha(batch)
        prod_dist = out.reshape(batch.shape[0], number_of_movies, 2)
       
        mu = self.scaling*torch.tanh(prod_dist[:, :, 0])
        sigma = torch.log(1 + torch.exp(prod_dist[:, :, 1]) + 0.01)
        
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

        probs = torch.stack([
            rating_1_prob, 
            rating_2_prob, 
            rating_3_prob, 
            rating_4_prob, 
            rating_5_prob
        ], dim=-1)

    
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
        y = y.type(torch.long)
        criterion = nn.NLLLoss()
        return criterion(yhat, y)

    def get_expected(self, pred):
        return 1.0*torch.exp(pred[:, 0]) +  2.0*torch.exp(pred[:, 1]) +  3.0*torch.exp(pred[:, 2]) +  4.0*torch.exp(pred[:, 3]) +  5.0*torch.exp(pred[:, 4])
    
    def get_expected_matrix(self, pred):
        return 1.0*torch.exp(pred[:, :, 0]) +  2.0*torch.exp(pred[:, :, 1]) +  3.0*torch.exp(pred[:,:,  2]) +  4.0*torch.exp(pred[:, :, 3]) +  5.0*torch.exp(pred[:, :, 4])
    


    def rmse(self, yhat, y):
        diff = (yhat - y) **2
        return torch.sqrt(torch.mean(diff))
    
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
      
     

    def training_step(self, batch, batch_idx):
        x, movie_mask, y = batch
        pred, mu, sigma = self(x)

        pred = pred[movie_mask]
        mu = mu[movie_mask]
        sigma = sigma[movie_mask]
        y = y[movie_mask]
        yhat = self.get_expected(pred)

        penalty = torch.mean(self.penalty_term(mu, sigma))
        nll = self.loss(pred, y-1)

        loss = nll + self.alpha*penalty

        rmse = self.rmse(yhat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_nll', nll, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_penalty', penalty, on_epoch=True, on_step=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, movie_mask, y = batch

        pred, mu, sigma = self(x)
    
        pred = pred[movie_mask]
        y = y[movie_mask]
        
        print('pred_val', pred.shape)
        yhat = self.get_expected(pred)
    
        nll = self.loss(pred, y -1)
        mse = self.rmse(y, yhat)
        self.log('val_loss', nll, on_epoch=True, on_step=True, prog_bar=True)
        self.log('val_rmse', mse, on_epoch=True, on_step=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        pred, mu, sigma = self(batch)
        res = self.get_expected_matrix(pred)
        return  res