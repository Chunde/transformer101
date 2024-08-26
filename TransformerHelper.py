import torch as tc
import torch.nn as nn
import Transformer
import InputEmbedding
import PositionalEncoding
import FeedForwardBlock
import ResidualConnection
import EncoderBlock
import DecoderBlock
import Encoder
import Decoder
import Transformer
import MultiHeadAttentionBlock


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

    sourcePosition = PositionalEncoding(d_model, sourceVocabSize, dropout)
    targetPosition = PositionalEncoding(d_model, targetVocabSize, dropout)

    encoderBlocks = []

    for _ in N:
        selfAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        feedForwardBlock = FeedForwardBlock(d_model, d_ff, dropout)

        encoderBlock = EncoderBlock(selfAttentionBlock, feedForwardBlock, dropout)
        encoderBlocks.append(encoderBlock)

    decoderBlocks = []
    for _ in N:
        selfAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        crossAttentionBlock = MultiHeadAttentionBlock(d_model, h, dropout)
        feedForwardBlock = FeedForwardBlock(d_model, d_ff, dropout)
        decoderBlock = DecoderBlock(
            selfAttentionBlock, crossAttentionBlock, feedForwardBlock, dropout
        )
        decoderBlocks.append(decoderBlock)

    encoder = Encoder(nn.ModuleList(encoderBlocks))
    decoder = Decoder(nn.ModuleList(decoderBlocks))

    projectLayer = projectLayer(d_model, targetVocabSize)

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