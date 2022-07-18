from turtle import forward
import torch 
import torch.nn as nn 
import pytorch_lightning as pl
number_of_users, number_of_movies = (10000, 1000)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_size, out_size, emb_size, num_heads, device) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        
        self.q = nn.Linear(in_size, emb_size, bias=False)
        self.k = nn.Linear(in_size, emb_size, bias=False)
        self.v = nn.Linear(in_size, emb_size)

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

    def __init__(self, emb_size, num_heads, num_layers, lr, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay


        self.mha = nn.ModuleList()
        
        self.mha.append(MultiHeadAttention(emb_size, emb_size, emb_size*num_heads, num_heads, device=self.device))
        for _ in range(num_layers-1):
           self.mha.append(MultiHeadAttention(emb_size, emb_size, emb_size*num_heads, num_heads, device=self.device))

        self.mha.append(MultiHeadAttention(emb_size, emb_size, emb_size*num_heads, num_heads, device=self.device))

        self.to_emb = nn.Linear(number_of_movies, emb_size)
        self.to_movie = nn.Linear(emb_size, number_of_movies)

    def forward(self, batch):
        out = self.to_emb(batch)
        for i in range(len(self.mha)):
            out = out + self.mha[i](out)
        
        return self.to_movie(out)
    
    def loss(self, yhat, y, movie_mask):
        """
        RMSE
        """
        diff = (yhat - y) **2
        return torch.sqrt(torch.mean(torch.masked_select(diff, movie_mask)))

    def configure_optimizers(self):

        opt =  torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer" : opt,
            "lr_scheduler" : torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: self.lr + epoch*(self.lr*0.1 - self.lr)/40 )
        }
    
    def training_step(self, batch, batch_idx):
        x, movie_mask, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y, movie_mask)
        self.log('train_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, movie_mask, y = batch

        yhat = self(x)
    
        loss = self.loss(yhat, y, movie_mask)
        self.log('val_rmse', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
