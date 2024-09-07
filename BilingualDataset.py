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
        # print(f"Source: {sourceText}")
        # print(f"Target: {targetText}")

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
        encoderMask = (encoderInput != self.padToken).unsqueeze(0).unsqueeze(0).int()
        # need to play with tensor view and slicing
        # lot of to be learnt, or at least be comfortable
        # with it and immediately know how to fix it
        decoderMask = (decoderInput != self.padToken).unsqueeze(0).int() & causalMask(
            decoderInput.size(0)
        )
        return {
            "encoderInput": encoderInput,  # sequence length
            "decoderInput": decoderInput,  # sequence length
            "encoderMask": encoderMask,
            "decoderMask": decoderMask,
            "label": label,
            "sourceText": sourceText,
            "targetText": targetText,
        }


def causalMask(size):
    # this function yield wrong shapes of mask that lead to error
    mask = tc.triu(tc.ones((1, size, size)), diagonal=1).type(tc.int)
    return mask == 0
