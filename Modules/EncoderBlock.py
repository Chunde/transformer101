import torch as tc
import torch.nn as nn
from . import MultiHeadAttentionBlock
from . import FeedForwardBlock
from . import ResidualConnection

class EncoderBlock(nn.Module):
    
    def __init__(self,
                 attentionBlock: MultiHeadAttentionBlock,
                 feedForward: FeedForwardBlock,
                 dropout: float)->None:
        
        super().__init__()
        self.attentionBlock = attentionBlock
        self.feedForward = feedForward
        self.residualConnection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # why we create to residue connection here?
        
    def forward(self, x, sourceMask):
        # question: so instantiate a instance call the init, but () operator will call forward? 22:54, 08/25/2024
        x = self.residualConnection[0](x, lambda x: self.attentionBlock(x, x, x, sourceMask))
        x = self.residualConnection[1](x, lambda x: self.feedForward(x))
        return x