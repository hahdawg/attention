import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, dropout=0.0):
        """
        Parameters
        ----------
        temperature: float
        dropout: float
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Parameters
        ----------
        q: Tensor(shape=(batch, n_head, num query steps, d_k))
        k: Tensor(shape=(batch, n_head, num input steps, d_k))
        v: Tensor(shape=(batch, n_head, num input steps, d_v))

        Returns
        -------
        (output, attention)
            output: Tensor(shape=(batch, n_head, len_q, len_k)
            attention: Tensor(shape=(batch, n_head, len_q, len_k)
        """
        # similarity.shape = (batch, n_head, num query steps, num input steps)
        similarity = torch.matmul(q, k.transpose(2, 3))/self.temperature

        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)

        # attention.shape = (batch, n_head, num query steps, num input steps)
        attention = torch.softmax(similarity, dim=-1)
        attention = self.dropout(attention)

        # output.shape = (batch, n_head, num query steps, num input steps)
        output = torch.matmul(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        """
        Parameters
        ----------
        n_head: int
        d_model: int
            Model inputs/query vectors will be like (batch_size, seq_len, d_model).
        d_k: int
            Dimension of key vector.
        d_v: int
            Dimension of value vector.
        dropout: float
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(in_features=d_model, out_features=n_head*d_k, bias=False)
        self.w_ks = nn.Linear(in_features=d_model, out_features=n_head*d_k, bias=False)
        self.w_vs = nn.Linear(in_features=d_model, out_features=n_head*d_v, bias=False)
        self.fc = nn.Linear(in_features=n_head*d_v, out_features=d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        Parameters
        ----------
        q: Tensor(shape=(batch_size, len_q, d_model))
        k: Tensor(shape=(batch_size, len_k, d_model))
        v: Tensor(shape=(batch_size, len_v, d_model))
        mask: Tensor(shape=(batch_size, len_q, len_k))

        Returns
        -------
        (output, attention)
            output: Tensor(shape=(batch_size, len_q, d_model))
            attention: Tensor(shape=(batch_size, n_head, len_q, len_k)
        """
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add dimension for heads

        q, attention = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2)  # q.shape = (batch_size, len_q, n_head, d_k)
        q = q.reshape(batch_size, len_q, -1)  # q.shape = (batch_size, len_q, n_head*d_k)
        q = self.fc(q)  # q.shape = (batch_size, len_q, d_model)
        q = self.dropout(q)
        q += residual
        q = self.layer_norm(q)
        return q, attention


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hidden, dropout):
        """
        Parameters
        ----------
        d_in: int
        d_hidden: int
        dropout: float
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_in, out_features=d_hidden)
        self.fc2 = nn.Linear(in_features=d_hidden, out_features=d_in)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=[batch_size, ..., d_in])

        Returns
        -------
        Tensor(shape=[batch_size, ..., d_in]
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(torch.relu(x))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoder(nn.Module):

    def __init__(self, d_model):
        """
        Parameters
        ----------
        d_model: int
        """
        super().__init__()
        self.d_model = d_model
        self._denom = 10000**(2.0*torch.arange(0, self.d_model)/(1.0*self.d_model))

    def forward(self, seq):
        """
        Parameters
        ----------
        seq: Tensor(shape=(batch, num steps, d_model)

        Returns
        -------
        Tensor(shape=(batch, num steps, d_model)
        """
        positions = torch.arange(seq.shape[1])
        seq_len = seq.shape[1]
        encoding = torch.zeros((seq_len, self.d_model))
        enc_input = positions.reshape(-1, 1) / self._denom
        encoding[:, ::2] = torch.sin(enc_input[:, ::2])
        encoding[:, 1::2] = torch.cos(enc_input[:, 1::2])
        encoded = seq + encoding
        return encoded


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        dropout
    ):
        """
        Parameters
        ----------
        d_model: int
        d_inner: int
        n_head: int
        droput: float
        """
        self.attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_model,
            d_v=d_model,
            dropout=dropout
        )
        self.pff = PositionwiseFeedForward(
            d_in=d_model,
            d_hidden=d_inner,
            dropout=dropout
        )

    def forward(self, encoder_input, mask=None):
        """
        Parameters
        ----------
        encoder_input: Tensor(shape=[batch, num_encode_steps, d_model])

        Returns
        -------
        (output, attention)
            output: Tensor(shape=[batch, num_encode_steps, d_model])
            attention: Tensor(shape=[batch, num_encode_steps, num_encode_steps])
        """
        output, attention = self.attention(
            q=encoder_input,
            k=encoder_input,
            v=encoder_input,
            mask=mask
        )
        output = self.pff(output)
        return output, attention


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        d_inner,
        d_k,
        d_v,
        n_head,
        dropout
    ):
        """
        Parameters
        ----------
        d_model: int
        d_inner: int
        d_k: int
        d_v: int
        n_head: int
        droput: int
        """
        self.decoder_decoder_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_model,
            d_v=d_model,
            dropout=dropout
        )
        self.decoder_encoder_attention = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.pff = PositionwiseFeedForward(
            d_in=d_model,
            d_hidden=d_inner,
            dropout=dropout
        )

    def forward(
        self,
        encoder_output,
        decoder_input,
        decoder_decoder_mask=None,
        decoder_encoder_mask=None
    ):
        """
        Parameters
        ----------
        encoder_output: Tensor(shape=[batch, num_encode_steps, d_model)
        decoder_input: Tensor(shape=[batch, num_decode_steps, d_model)
        decoder_decoder_mask: Tensor(shape=[batch, num_decode_steps, num_decode_steps])
        decoder_encoder_mask: Tensor(shape=[batch, num_decode_steps, num_encode_steps])

        Returns
        -------
        (decoder_output, decoder_decoder_attention, decoder_encoder_attention)
            decoder_output: Tensor(shape=[batch, num_decode_steps, d_model])
            decoder_decoder_attention: Tensor(shape=[batch, num_decode_steps, num_decode_steps])
            decoder_encoder_attention: Tensor(shape=[batch, num_decode_steps, num_encode_steps])
        """
        decoder_output, decoder_decoder_attention = self.decoder_decoder_attention(
            q=decoder_input,
            k=decoder_input,
            v=decoder_input,
            mask=decoder_decoder_mask
        )
        decoder_output, decoder_encoder_attention = self.decoder_encoder_attention(
            q=decoder_output,
            k=encoder_output,
            v=encoder_output,
            mask=decoder_encoder_mask
        )
        decoder_output = self.pff(decoder_output)
        return decoder_output, decoder_decoder_attention, decoder_encoder_attention


class Encoder(nn.Module):

    def __init__(
        self,
        num_src_vocab,
        embedding_dim,
        num_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        num_positions=200
    ):
        self.src_word_embedding = nn.Embedding(
            num_embeddings=num_src_vocab,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                dropout=dropout
            ) for _ in range(num_layers)

        ])
        self.positional_encoder = PositionalEncoder(d_model=d_model)

    def forward(self, encoder_input, mask, return_attentions=False):
        output = self.src_word_embedding(encoder_input)
        output = self.positional_encoder(output)
        output = self.dropout(output)
        attentions = []
        for layer in self.layer_stack:
            output, attention = layer(output, mask=mask)
            attentions.append(attention)

        if return_attentions:
            return output, attentions

        return output


class Decoder(nn.Module):

    def __init__(
        self,
        num_dest_vocab,
        embedding_dim,
        num_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        num_positions=200
    ):
        self.src_word_embedding = nn.Embedding(
            num_embeddings=num_dest_vocab,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                d_k=d_k,
                d_v=d_v,
                n_head=n_head,
                dropout=dropout
            ) for _ in range(num_layers)

        ])
        self.positional_encoder = PositionalEncoder(d_model=d_model)

    def forward(
        self,
        encoder_output,
        decoder_input,
        decoder_decoder_mask=None,
        decoder_encoder_mask=None,
        return_attentions=False
    ):
        """
        Parameters
        ----------
        encoder_output: Tensor(shape=[batch, num encoder steps, d_model)
        decoder_input: Tensor(shape=[batch, num decoder steps, d_model)
        decoder_decoder_mask: Tensor(shape=[batch, num decoder steps, num decoder steps])
        decoder_encoder_mask: Tensor(shape=[batch, num decoder steps, num encoder steps])
        return_attentions: bool

        Returns
        -------
        (decoder_output, decoder_decoder_attention, decoder_encoder_attention)
        """
        output = self.src_word_embedding(decoder_input)
        output = self.positional_encoder(output)
        output = self.dropout(output)
        attentions = []
        for layer in self.layer_stack:
            output, attention = layer(
                encoder_output=encoder_output,
                decoder_input=decoder_input,
                decoder_decoder_mask=decoder_decoder_mask,
                decoder_encoder_mask=decoder_encoder_mask
            )
            attentions.append(attention)

        if return_attentions:
            return output, attentions

        return output
