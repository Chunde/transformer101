import torch as tc
import torch.nn as nn
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .FeedForwardBlock import FeedForwardBlock
from .ResidualConnection import ResidualConnection


class DecoderBlock(nn.Module):

    def __init__(
        self,
        selfAttentionBlock: MultiHeadAttentionBlock,
        crossAttentionBlock: MultiHeadAttentionBlock,
        feedForwardBlock: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.selfAttentionBlock = selfAttentionBlock
        self.crossAttentionBlock = crossAttentionBlock
        self.feedForwardBlock = feedForwardBlock
        self.residualConnection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoderOutput, sourceMask, targetMask):
        x = self.residualConnection[0](
            x, lambda x: self.selfAttentionBlock(x, x, x, targetMask)
        )
        x = self.residualConnection[1](
            x,
            lambda x: self.crossAttentionBlock(
                x, encoderOutput, encoderOutput, sourceMask
            ),
        )
        x = self.residualConnection[2](x, lambda x: self.feedForwardBlock(x))
