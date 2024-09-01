import torch.nn as nn

# x dims = (batch size, sequence length, d_model)
# class InputEmbedding(nn.Module):
    
#     def ___init__(self, d_model:int, vocab_size:int):
#         super().__init__()
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)
    
#     def forward(self, x):
#         return self.embedding(x)*self.d_model**0.5
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
         return self.embedding(x)*self.d_model**0.5