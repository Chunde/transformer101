import torch as tc
import torch.nn as nn
import math
   
# x dims = (batch size, sequence length, d_model)   
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear_dim1 = nn.Linear(d_model, d_ff) #d_model = 512, d_ff = 2048 per original paper
        self.dropout = nn.Dropout(dropout)
        
        self.linear_dim2 = nn.Linear(d_ff, d_model) #w2 and b2
        
    def forward(self, x):
        return self.linear_dim2(self.dropout(tc.relu(self.linear_dim1(x)))) # check how dimension works here