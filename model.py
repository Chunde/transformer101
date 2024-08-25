import torch as tc
import torch.nn as nn
import math




class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter(tc.ones(1))
        self.bias = nn.parameter(tc.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear_dim1 = nn.Linear(d_model, d_ff) #d_model = 512, d_ff = 2048 per original paper
        self.dropout = nn.Dropout(dropout)
        
        self.linear_dim2 = nn.Linear(d_ff, d_model) #w2 and b2
        
    def forward(self, x):
        # x dims = (batch size, sequence length, d_model)
        return self.linear_dim2(self.dropout(tc.relu(self.linear_dim1(x))))
    
class MultiHeadAttentionBlock(nn.Module)    :
    
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % h == 0, "dimension of embedding should be divisible by number of headers h"
        self.d_k = self.d_model // h
        self.w_q = nn.Linear(d_model, d_model) # query
        self.w_k = nn.Linear(d_model, d_model) # key
        self.w_v = nn.Linear(d_model, d_model) # value
        
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        
    def forward(self, q, k, v, mask)   :
        q_prime = self.w_q(q)
        k_prime = self.w_k(k)
        v_prime = self.w_v(v)
        
        # next, we need to split these metrics into different header metrics
        # (batch, sequence, dim of embedding) -> (batch, sequence, h, d_k) - > (batch, h, seq, dk)
        q_prime = q_prime.view(q_prime.shape[0], q_prime.shape[1], self.h, self.d_k).transpose(1,2)
        k_prime = k_prime.view(k_prime.shape[0], k_prime.shape[1], self.h, self.d_k).transpose(1,2)
        v_prime = q_prime.view(v_prime.shape[0], v_prime.shape[1], self.h, self.d_k).transpose(1,2)
        
        # batch,h seq, dk --> batch, h, seq, seq
        # this operation should be very cautious--23:22
        attention_score = (q_prime @ k_prime.transpose(-1,-2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        self.attention_score = attention_score.softmax(dim = -1)
        # (batch, h, seq, seq) @ (batch,h, seq, d_k) -> (batch, h, seq, dk)
        x = attention_score @v_prime
        
        