"""
model.py — The heart of the project: a flexible character-level RNN (LSTM/GRU/RNN)

I wanted something that could switch between LSTM, GRU, or plain RNN without
rewriting everything. Returns raw logits so training with CrossEntropyLoss is straightforward.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CharRNN(nn.Module):
    """
    Simple but effective character-level RNN.

    I went with stacked layers + dropout between them. Works surprisingly well
    on Shakespeare even with default settings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        model_type: str = "LSTM",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.upper()

        # Embedding layer — turns char indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The recurrent stack
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_kwargs = dict(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs, nonlinearity="tanh")

        # Final projection + dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Weight initialization that actually works well in practice."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # Orthogonal init for recurrent weights — my favorite trick for RNN stability
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Standard forward pass.

        x: (batch, seq_len) of char indices
        hidden: previous state or None
        Returns: logits (batch, seq_len, vocab), updated hidden
        """
        emb = self.embedding(x)
        out, h = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, h

    def init_hidden(self, batch_size: int, device: torch.device):
        """Fresh zero hidden state. LSTM gets both h and c."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.model_type == "LSTM":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def detach_hidden(self, hidden):
        """Detach for truncated BPTT so we don't backprop forever."""
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
