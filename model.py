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
    
    
    