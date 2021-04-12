import unittest
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
        self.query_xf = nn.Linear(dmodel, dk, False)
        self.key_xf = nn.Linear(dmodel, dk, False)
        self.value_xf = nn.Linear(dmodel, dv, False)
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
        Q = self.query_xf(x)  # (batch, seq_len, dk)
        K = self.key_xf(x)  # (batch, seq_len, dk)
        V = self.value_xf(x)  # (batch, seq_len, dv)
        KT = torch.transpose(K, 2, 1)
        E = torch.matmul(Q, KT) / self.similarity_norm  # (batch, seq_len, seq_len)
        E = self.causal_mask(E)
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
        y = torch.cat([head(x) for head in self.heads], dim=-1)
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


# pylint: disable=R0902
class TransformTester(unittest.TestCase):
    # TODO: Move these somewhere suitable
    def setUp(self):
        self.dmodel = 2
        self.num_heads = 3
        self.num_layers = 5
        self.dim_feedforward = 7
        self.batch_size = 11
        self.seq_len = 13
        self.dk = 17
        self.dv = 19
        self.dropout = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x = torch.randn((self.batch_size, self.seq_len, self.dmodel)).to(self.device)

    def test_causal_mask(self):
        dim = 2
        x = torch.ones((dim, dim))
        masked = CausalMask()(x)
        self.assertEqual(masked[0, 0], 1)
        self.assertTrue(masked[0, 1] < -999999999)
        self.assertEqual(masked[1, 1], 1)
        self.assertEqual(masked[1, 1], 1)

    def test_attention_layer(self):
        layer = AttentionLayer(
            dmodel=self.dmodel,
            dk=self.dk,
            dv=self.dv,
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dv))

    def test_mh_attention_layer(self):
        layer = MultiHeadAttentionLayer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            dk=self.dk,
            dv=self.dv
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_transformer_layer(self):
        layer = TransformerLayer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_positionwise_ff(self):
        ff = PositionwiseFeedForward(
            dmodel=self.dmodel,
            dhidden=self.dim_feedforward
        ).to(self.device)
        y = ff(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_positional_encoder(self):
        pe = PositionalEncoder().to(self.device)

        x_even = torch.zeros((self.batch_size, self.seq_len, self.dmodel)).to(self.device)
        self.assertEqual(pe(x_even).shape, x_even.shape)

        x_odd = torch.zeros((self.batch_size, self.seq_len + 1, self.dmodel)).to(self.device)
        self.assertEqual(pe(x_odd).shape, x_odd.shape)

    def test_transformer(self):
        tm = Transformer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout
        ).to(self.device)
        y = tm(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_language_model(self):
        embedding_size = 2
        num_heads = 3
        num_layers = 5
        dim_feedforward = 7
        dim_feedforward = 11
        seq_len = 13
        batch = 17
        vocab_size = 19
        lm = LanguageModel(
            embedding_size=embedding_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout
        ).to(self.device)
        x = torch.randint(0, vocab_size, size=(batch, seq_len)).to(self.device)
        y = lm(x)
        self.assertEqual(y.shape, (batch, seq_len, vocab_size))


if __name__ == "__main__":
    unittest.main()
