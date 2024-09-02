import torch as tc
import torch.nn as nn
from . import Transformer
from .InputEmbedding import InputEmbedding
from .PositionalEncoding import PositionalEncoding
from .FeedForwardBlock import FeedForwardBlock
from .ResidualConnection import ResidualConnection
from .EncoderBlock import EncoderBlock
from .DecoderBlock import DecoderBlock
from .Encoder import Encoder
from .Decoder import Decoder
from .Transformer import Transformer
from .MultiHeadAttentionBlock import MultiHeadAttentionBlock
from .ProjectLayer import ProjectLayer

def createTranslationTransformer(
    sourceVocabSize: int,
    targetVocabSize: int,
    sourceSequenceLen: int,
    targetSequenceLen: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> None:
    sourceEmbedding = InputEmbedding(d_model, sourceVocabSize)
    targetEmbedding = InputEmbedding(d_model, targetVocabSize)

    sourcePosition = PositionalEncoding(d_model, sourceSequenceLen, dropout)
    targetPosition = PositionalEncoding(d_model, targetSequenceLen, dropout)

    encoderBlocks = []

    for _ in range(N):
        selfAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        feedForwardBlock = FeedForwardBlock(d_model, d_ff, dropout)

        encoderBlock = EncoderBlock(selfAttentionBlock, feedForwardBlock, dropout)
        encoderBlocks.append(encoderBlock)

    decoderBlocks = []
    for _ in range(N):
        selfAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        crossAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        feedForwardBlock = FeedForwardBlock(d_model, d_ff, dropout)
        decoderBlock = DecoderBlock(
            selfAttentionBlock, crossAttentionBlock, feedForwardBlock, dropout
        )
        decoderBlocks.append(decoderBlock)

    encoder = Encoder(nn.ModuleList(encoderBlocks))
    decoder = Decoder(nn.ModuleList(decoderBlocks))

    projectLayer = ProjectLayer(d_model, targetVocabSize)

    transformer = Transformer(
        encoder,
        decoder,
        sourceEmbedding,
        sourcePosition,
        targetEmbedding,
        targetPosition,
        projectLayer,
    )

    # Initialize parameters
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return transformer