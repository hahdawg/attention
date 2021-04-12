import unittest

import torch

import attention.model as am


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
        self.dropout = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x = torch.randn((self.batch_size, self.seq_len, self.dmodel)).to(self.device)

    def test_causal_mask(self):
        dim = 2
        x = torch.ones((dim, dim))
        masked = am.CausalMask()(x)
        self.assertEqual(masked[0, 0], 1)
        self.assertTrue(masked[0, 1] < -999999999)
        self.assertEqual(masked[1, 1], 1)
        self.assertEqual(masked[1, 1], 1)

    def test_attention_layer(self):
        layer = am.AttentionLayer(
            dmodel=self.dmodel,
            dk=self.dk,
            dv=self.dv,
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dv))

    def test_mh_attention_layer(self):
        layer = am.MultiHeadAttentionLayer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            dk=self.dk,
            dv=self.dv
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_transformer_layer(self):
        layer = am.TransformerLayer(
            dmodel=self.dmodel,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        y = layer(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_positionwise_ff(self):
        ff = am.PositionwiseFeedForward(
            dmodel=self.dmodel,
            dhidden=self.dim_feedforward
        ).to(self.device)
        y = ff(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.dmodel))

    def test_positional_encoder(self):
        pe = am.PositionalEncoder().to(self.device)

        x_even = torch.zeros((self.batch_size, self.seq_len, self.dmodel)).to(self.device)
        self.assertEqual(pe(x_even).shape, x_even.shape)

        x_odd = torch.zeros((self.batch_size, self.seq_len + 1, self.dmodel)).to(self.device)
        self.assertEqual(pe(x_odd).shape, x_odd.shape)

    def test_transformer(self):
        tm = am.Transformer(
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
        lm = am.LanguageModel(
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
