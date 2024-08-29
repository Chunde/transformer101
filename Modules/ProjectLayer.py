import torch as tc
import torch.nn as nn
import LayerNormalization

class ProjectLayer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return tc.log_softmax(self.projection(x), dim=-1)