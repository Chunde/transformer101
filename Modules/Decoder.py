import torch as tc
import torch.nn as nn
from . import LayerNormalization

class Decoder(nn.Module):
    
    def __init__(self, layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoderOutput, sourceMask, targetMask):
        for layer in self.layers:
            x = layer(x, encoderOutput, sourceMask, targetMask)
            
        return self.norm(x)