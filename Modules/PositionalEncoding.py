import torch as tc
import torch.nn as nn
import math

# x dims = (batch size, sequence length, d_model)
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_size:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_size = seq_size
        self.dropout = nn.Dropout(dropout)
        # so the position encoder carries information about 
        # relative position inside a training sequence (this max length)
        # In the YT video, is pick a constant of 10000 in the denominator
        # which has not to be that, as long as it is no less than the
        # max sequence size
        pe = tc.zeros(seq_size, d_model)
        # let me think how to compute these number in simple way, it is simple
        # mathematically. here is naive way to do it. we will come back later
        # when start to doing unit test--12:20 08/24/2024
        for pos in range(seq_size):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = math.sin(pos / math.pow(10000.0, float(i)/float(d_model)))
                else:
                    pe[pos][i] = math.cos(pos / math.pow(10000.0, float(i - 1)/float(d_model)))
                    
        pe = pe.unsqueeze(0) # 
        self.register_buffer('pe', pe) # save the pe as part of model
    
    def forward(self, x):
        x = x + (self.pe[:, x.shape[1],:]).require_grad_(False) # this should be checked again
        