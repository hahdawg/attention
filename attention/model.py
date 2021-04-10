import unittest
import math

import torch
from torch import Tensor
import torch.nn as nn


class AttentionLayer(nn.Module):

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
        self.softmax = nn.Softmax(dim=2)

    @staticmethod
    def causal_mask(x: Tensor) -> Tensor:
        mask = -torch.triu(float("inf")*torch.ones_like(x), diagonal=1)
        return x + mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, nx, dmodel))

        Returns
        -------
        Tensor(size=(batch, nx, dv))
        """
        Q = self.query_xf(x)  # (batch, nx, dk)
        K = self.key_xf(x)  # (batch, nx, dk)
        V = self.value_xf(x)  # (batch, nx, dv)
        KT = torch.transpose(K, 2, 1)
        E = torch.matmul(Q, KT) / self.similarity_norm  # (batch, nx, nx)
        E = self.causal_mask(E)
        A = self.softmax(E)  # (batch, nx, nx)
        y = torch.matmul(A, V)  # (batch, nx, dv)
        return y


class MultiHeadAttentionLayer(nn.Module):

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
        x: Tensor(size=(batch, nx, dmodel))

        Returns
        -------
        Tensor(size=(batch, nx, dmodel))
        """
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.output_layer(y)
        return y


class PositionwiseFeedForward(nn.Module):

    def __init__(self, dmodel: int, dhidden: int):
        super().__init__()
        self.linear_input = nn.Linear(dmodel, dhidden)
        self.linear_output = nn.Linear(dhidden, dmodel)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, nx, dmodel))

        Returns
        -------
        Tensor(size=(batch, nx, dmodel))
        """
        x = self.linear_input(x)
        x = self.relu(x)
        return self.linear_output(x)


class TransformerLayer(nn.Module):

    def __init__(
        self,
        dmodel: int,
        num_heads: int,
        dim_feedforward: int
    ):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(
            dmodel=dmodel,
            num_heads=num_heads
        )
        self.layer_norm_attention = nn.LayerNorm(dmodel, dim_feedforward)

        self.feedforward = PositionwiseFeedForward(
            dmodel=dmodel,
            dhidden=dim_feedforward
        )
        self.layer_norm_feedforward = nn.LayerNorm(dmodel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, nx, dmodel))

        Returns
        -------
        Tensor(size=(batch, nx, dmodel))
        """
        x = self.attention(x) + x
        x = self.layer_norm_attention(x)
        x = self.feedforward(x) + x
        x = self.layer_norm_feedforward(x)
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        dmodel: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                dmodel=dmodel,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, nx, dmodel))

        Returns
        -------
        Tensor(size=(batch, nx, dmodel))
        """
        for layer in self.layers:
            x = layer(x)
        return x


class LanguageModel(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transfomer = Transformer(
            dmodel=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        self.output_layer = nn.Linear(embedding_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch, nx, embedding_size))

        Returns
        -------
        Tensor(size=(batch, nx, embedding_size))
        """
        x = self.embedding(x)
        x = self.transfomer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


# pylint: disable=R0902
class TransformTester(unittest.TestCase):

    def setUp(self):
        self.dmodel = 2
        self.num_heads = 3
        self.num_layers = 5
        self.dim_feedforward = 7
        self.batch_size = 11
        self.seq_len = 13
        self.dk = 17
        self.dv = 19
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x = torch.randn((self.batch_size, self.seq_len, self.dmodel)).to(self.device)

    def test_causal_mask(self):
        dim = 2
        x = torch.ones((dim, dim))
        masked = AttentionLayer.causal_mask(x)
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

    def test_transformer(self):
        tm = Transformer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            device=self.device
        )
        y = tm(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))


if __name__ == "__main__":
    unittest.main()
