from turtle import forward
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup


class MultiHeadAttention(nn.Module):
    def __init__(self, in_size, out_size, emb_size, num_heads, 
        hidden_size_q_1, 
        hidden_size_q_2,
        hidden_size_v_1, 
        hidden_size_v_2,
        hidden_size_k_1, 
        hidden_size_k_2, device) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        
        self.q = nn.Sequential(
             nn.Linear(in_size, hidden_size_q_1),
             nn.ReLU(),
             nn.Linear(hidden_size_q_1, hidden_size_q_2),
             nn.ReLU(),
             nn.Linear(hidden_size_q_2, emb_size)
        )

        self.v = nn.Sequential(
             nn.Linear(in_size, hidden_size_v_1),
             nn.ReLU(),
             nn.Linear(hidden_size_v_1, hidden_size_v_2),
             nn.ReLU(),
             nn.Linear(hidden_size_v_2, emb_size)
        )

        self.k = nn.Sequential(
             nn.Linear(in_size, hidden_size_k_1),
             nn.ReLU(),
             nn.Linear(hidden_size_k_1, hidden_size_k_2),
             nn.ReLU(),
             nn.Linear(hidden_size_k_2, emb_size)
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
        hidden_size_q_1, 
        hidden_size_q_2,
        hidden_size_v_1, 
        hidden_size_v_2,
        hidden_size_k_1, 
        hidden_size_k_2, num_heads, lr, weight_decay, 
        warmup_steps=None,
        total_steps=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.mha = MultiHeadAttention(number_of_movies, number_of_movies, emb_size*num_heads, num_heads, 
                hidden_size_q_1, 
                hidden_size_q_2,
                hidden_size_v_1, 
                hidden_size_v_2,
                hidden_size_k_1, 
                hidden_size_k_2,
                device=self.device)

    def forward(self, batch):
        return self.mha(batch)
    
    def loss(self, yhat, y, movie_mask):
        
        diff = (yhat - y) **2
        return torch.sqrt(torch.mean(torch.masked_select(diff, movie_mask)))

    # def loss(self, yhat, y, movie_mask):
    #     """
    #     RMSE
    #     """
    #     diff = torch.abs(yhat - y)
    #     return torch.mean(torch.masked_select(diff, movie_mask))
    
    

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        opt = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        return opt
        # optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=self.lr)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.total_steps)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": lr_scheduler,
        #         "monitor": "val_rmse",
        #     },
        # }
    def training_step(self, batch, batch_idx):
        x, movie_mask, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y, movie_mask)
        #rmse = self.rmse(yhat, y, movie_mask)
        self.log('train_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        #self.log('train_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, movie_mask, y = batch

        yhat = self(x)
    
        loss = self.loss(yhat, y, movie_mask)
        #rmse = self.rmse(yhat, y, movie_mask)
        self.log('val_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        #self.log('val_rmse', rmse, on_epoch=True, on_step=True, prog_bar=True)
        return loss
