import math

import torch
from torch import Tensor
import torch.nn as nn


class CausalMask(nn.Module):
    """
    Given an square tensor x, add -inf to upper triangular elements (excluding diagonal).
    """
    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=R0201
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, seq_len))

        Returns
        -------
        Tensor(size=(batch, seq_len, seq_len))
        """
        # (seq_len, seq_len)
        mask = -torch.triu(float("inf")*torch.ones_like(x), diagonal=1).to(x.device)
        return x + mask


class AttentionLayer(nn.Module):
    """
    Standard (self) attention layer. Should be applied to embeddings.
    """
    def __init__(
        self,
        dmodel: int,
        dk: int,
        dv: int
    ):
        super().__init__()
        self.similarity_norm = math.sqrt(dk)
        self.query_xform = nn.Linear(dmodel, dk, False)
        self.key_xform = nn.Linear(dmodel, dk, False)
        self.value_xform = nn.Linear(dmodel, dv, False)
        self.causal_mask = CausalMask()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel), dtype=float)

        Returns
        -------
        Tensor(size=(batch, seq_len, dv))
        """
        Q = self.query_xform(x)  # (batch, seq_len, dk)
        K = self.key_xform(x)  # (batch, seq_len, dk)
        V = self.value_xform(x)  # (batch, seq_len, dv)
        KT = torch.transpose(K, 2, 1)  # (batch, dk, seq_len)
        E = torch.matmul(Q, KT) / self.similarity_norm  # (batch, seq_len, seq_len)
        E = self.causal_mask(E)
        # E[i, j] = similarity between Q[i] and K[j], so want softmax over dim = -1.
        A = self.softmax(E)  # (batch, seq_len, seq_len)
        y = torch.matmul(A, V)  # (batch, seq_len, dv)
        return y


class MultiHeadAttentionLayer(nn.Module):
    """
    Standard (self) multihead attention layer. Should be applied to embeddings.
    """
    def __init__(
        self,
        dmodel: int,
        num_heads: int,
        dk: int = None,
        dv: int = None
    ):
        super().__init__()
        dk = dk or dmodel
        dv = dv or dmodel
        self.heads = nn.ModuleList([
            AttentionLayer(dmodel=dmodel, dk=dk, dv=dv)
            for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(num_heads*dv, dmodel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel), dtype=float)

        Returns
        -------
        Tensor(size=(batch, seq_len, dmodel))
        """
        y = torch.cat([head(x) for head in self.heads], dim=-1)  # batch x seq_len x (num_heads*dv)
        y = self.output_layer(y)
        return y


class PositionwiseFeedForward(nn.Module):
    """
    Feed forward network for transformer block.
    """
    def __init__(self, dmodel: int, dhidden: int):
        super().__init__()
        self.linear_input = nn.Linear(dmodel, dhidden)
        self.linear_output = nn.Linear(dhidden, dmodel)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel))

        Returns
        -------
        Tensor(size=(batch, seq_len, dmodel))
        """
        x = self.linear_input(x)
        x = self.relu(x)
        return self.linear_output(x)


class TransformerLayer(nn.Module):
    """
    Self-attentive transformer layer. Should be applied to embeddings.
    """
    def __init__(
        self,
        dmodel: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(
            dmodel=dmodel,
            num_heads=num_heads
        )
        self.layer_norm_attention = nn.LayerNorm(dmodel)

        self.feedforward = PositionwiseFeedForward(
            dmodel=dmodel,
            dhidden=dim_feedforward
        )
        self.layer_norm_feedforward = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel), dtype=float)

        Returns
        -------
        Tensor(size=(batch, seq_len, dmodel))
        """
        x = self.dropout(self.attention(x)) + x
        x = self.layer_norm_attention(x)
        x = self.dropout(self.feedforward(x)) + x
        x = self.layer_norm_feedforward(x)
        return x


class Transformer(nn.Module):
    """
    Transformer model. Consists of transformer layers. Should be applied to embeddings.
    """
    def __init__(
        self,
        dmodel: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout_input: float,
        dropout_hidden: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                dmodel=dmodel,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout_hidden
            )
            for _ in range(num_layers)
        ])
        self.dropout_input = nn.Dropout(p=dropout_input)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel), dtype=float).

        Returns
        -------
        Tensor(size=(batch, seq_len, dmodel))
        """
        x = self.dropout_input(x)
        for layer in self.layers:
            x = layer(x)
        return x


class PositionalEncoder(nn.Module):
    """
    Adds positional information to an embedding tensor x.
    """
    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=R0201
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, dmodel), dtype=float)

        Returns
        -------
        Tensor(size=(batch, seq_len, dmodel))
        """
        _, seq_len, dmodel = x.shape
        norm_factor = (1 / 10_000 ** (2*torch.arange(dmodel)/dmodel)).reshape(-1, 1)
        evens = torch.sin(norm_factor * torch.arange(0, seq_len, 2))  # dmodel x (seq_len / 2)
        # Need same number of odd and even terms. If seq_len is odd, there will be more evens than
        # odds, so add another odd term for dstack.
        if seq_len % 2 == 0:
            num_odds = seq_len
        else:
            num_odds = seq_len + 1
        odds = torch.cos(norm_factor * torch.arange(1, num_odds, 2))  # dmodel x (seq_len / 2)
        # Reason for [:, :seq_len] at end: If seq_len is odd, we have one too many odds,
        # so drop the last one.
        embedding = torch.dstack((evens, odds)).reshape(dmodel, -1)[:, :seq_len]  # dmodel x seq_len
        embedding = embedding.to(x.device)
        return x + embedding.T


class LanguageModel(nn.Module):
    """
    Language model for next-word prediction.
    """
    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout_input: float = 0.1,
        dropout_hidden: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = PositionalEncoder()
        self.transfomer = Transformer(
            dmodel=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, seq_len, vocab_size), dtype=int)

        Returns
        -------
        Tensor(size=(batch, seq_len, vocab_size))
            Return logits (NOT probabilities).
        """
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transfomer(x)
        x = self.output_layer(x)
        return x
