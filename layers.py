"""https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb"""
import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=100,
        encoder_version="ReZero",
    ):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        if encoder_version == "ReZero":
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        hid_dim,
                        n_heads,
                        pf_dim,
                        device=self.device,
                        dropout=dropout,
                        use_layer_norm=False,
                        init_resweight=0,
                        resweight_trainable=True,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif encoder_version == "pre":
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        hid_dim,
                        n_heads,
                        pf_dim,
                        device=self.device,
                        dropout=dropout,
                        use_layer_norm="pre",
                        init_resweight=1,
                        resweight_trainable=False,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif encoder_version == "post":
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        hid_dim,
                        n_heads,
                        pf_dim,
                        device=self.device,
                        dropout=dropout,
                        use_layer_norm="post",
                        init_resweight=1,
                        resweight_trainable=False,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        device,
        dropout=0.1,
        use_layer_norm=False,
        init_resweight=0,
        resweight_trainable=True,
    ):
        super().__init__()

        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(
            torch.Tensor([init_resweight]), requires_grad=resweight_trainable
        )

        self.self_attn = MultiheadAttention(
            embed_dim=hid_dim, num_heads=n_heads, dropout=dropout, device=device, batch_first=True
        )

        self.linear1 = Linear(hid_dim, pf_dim)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(pf_dim, hid_dim)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm != False:
            self.norm1 = LayerNorm(hid_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask):
        src2 = src
        if self.use_layer_norm == "pre":
            src2 = self.norm1(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask)[0]
        # Apply the residual weight to the residual connection. This enables ReZero.
        src2 = self.resweight * src2
        src2 = self.dropout1(src2)
        if self.use_layer_norm == False:
            src = src + src2
        elif self.use_layer_norm == "pre":
            src = src + src2
        elif self.use_layer_norm == "post":
            src = self.norm1(src + src2)
        src2 = src
        if self.use_layer_norm == "pre":
            src2 = self.norm1(src2)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src2))))
        src2 = self.resweight * src2
        src2 = self.dropout2(src2)
        if self.use_layer_norm == False:
            src = src + src2
        elif self.use_layer_norm == "pre":
            src = src + src2
        elif self.use_layer_norm == "post":
            src = self.norm1(src + src2)
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, resweight=1):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=100,
    ):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
