import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(
        self,
        dataset,
        sourceTokenizer,
        targetTokenizer,
        sourceLang,
        targetLang,
        sequenceLen,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.sourceTokenizer = sourceTokenizer
        self.targetTokenizer = targetTokenizer
        self.sourceLang = sourceLang
        self.targetLang = targetLang
        self.sosToken = tc.tensor(
            [sourceTokenizer.token_to_id("[SOS]")], dtype=tc.int64
        )
        self.eosToken = tc.tensor(
            [sourceTokenizer.token_to_id("[EOS]")], dtype=tc.int64
        )
        self.padToken = tc.tensor(
            [sourceTokenizer.token_to_id("[PAD]")], dtype=tc.int64
        )
        self.sequenceLen = sequenceLen

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sourceTargetPair = self.dataset[index]
        sourceText = sourceTargetPair["translation"][self.sourceLang]
        targetText = sourceTargetPair["translation"][self.targetLang]

        encoderInputTokens = self.sourceTokenizer.encode(sourceText).ids
        decoderInputTokens = self.targetTokenizer.encode(targetText).ids

        encoderPaddingTokensNum = self.sequenceLen - len(encoderInputTokens) - 2
        decoderPaddingTokensNum = (
            self.sequenceLen - len(decoderInputTokens) - 1
        )  # no need ot EOS

        if encoderPaddingTokensNum < 0 or decoderPaddingTokensNum < 0:
            raise ValueError("Sequence length is not big enough")

        encoderInput = tc.cat(
            [
                self.sosToken,
                tc.tensor(encoderInputTokens, dtype=tc.int64),
                self.eosToken,
                tc.tensor([self.padToken] * encoderPaddingTokensNum, dtype=tc.int64),
            ]
        )

        decoderInput = tc.cat(
            [
                self.sosToken,
                tc.tensor(decoderInputTokens, dtype=tc.int64),
                tc.tensor([self.padToken] * decoderPaddingTokensNum, dtype=tc.int64),
            ]
        )

        label = tc.cat(
            [
                tc.tensor(decoderInputTokens, dtype=tc.int64),
                self.eosToken,
                tc.tensor([self.padToken] * decoderPaddingTokensNum, dtype=tc.int64),
            ]
        )

        assert encoderInput.size(0) == self.sequenceLen
        assert decoderInput.size(0) == self.sequenceLen
        assert label.size(0) == self.sequenceLen

        return {
            "encoderInput": encoderInput,  # sequence length
            "decoderInput": decoderInput,  # sequence length
            "encoderMask": (encoderInput != self.padToken)
            .unsqueeze(0)   # need to play with tensor view and slicing
            .unsqueeze(0)   # lot of to be learnt, or at least be comfortable
            .int(),         # with it and immediately know how to fix it
            "decoderMask": (decoderInput != self.padToken).unsqueeze(0).int()
            & causalMask(decoderInput.size(0)),
            "label": label,
            "sourceText": sourceText,
            "targetText": targetText,
        }


def causalMask(size):
    mask = tc.triu(tc.ones(size, size), diagonal=1).type(tc.int)
    return mask == 0
