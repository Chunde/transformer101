import torch as tc
import torch.nn as nn
import LayerNormalization
import Encoder
import Decoder
import InputEmbedding
import PositionalEncoding
import ProjectLayer


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sourceEmbedding: InputEmbedding,
        sourcePosition: PositionalEncoding,
        targetEmbedding: InputEmbedding,
        targetPosition: PositionalEncoding,
        projectLayer: ProjectLayer,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.sourceEmbedding = sourceEmbedding
        self.targetEmbedding = targetEmbedding
        self.sourcePosition = sourcePosition
        self.targetPosition = targetPosition
        self.projectionLayer = projectLayer

    def encode(self, source, sourceMask):
        source = self.sourceEmbedding(source)
        source = self.sourcePosition(source)
        return self.encoder(source, sourceMask)

    def decode(self, encoderOutput, sourceMask, target, targetMask):
        target = self.targetEmbedding(target)
        target = self.targetPosition(target)
        return self.decoder(target, encoderOutput, sourceMask, targetMask)

    def project(self, x):
        return self.projectionLayer(x)
    