import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.masking import OffDiagMask


class HODA(nn.Module):
    def __init__(self, num_layers, label_len, enc_in, d_model, dropout=0.1):
        super(HODA, self).__init__()
        self.num_layers = num_layers
        ODA_layers = [ODA_layer(label_len, i, enc_in, d_model, dropout=dropout)
                      for i in range(self.num_layers)]
        self.ODA_layers = nn.ModuleList(ODA_layers)

    def forward(self, x):
        input_x = x.clone()
        output = 0
        for num in range(self.num_layers):
            input_x, out = self.ODA_layers[num](input_x)
            output += out
        return output


class ODA_layer(nn.Module):
    def __init__(self, label_len, num_layers, enc_in, d_model, dropout=0.1):
        super(ODA_layer, self).__init__()
        self.num_layers = num_layers
        self.label_len = label_len
        self.enc_in = enc_in
        self.downsampling = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.upsampling = nn.AdaptiveAvgPool2d((enc_in, label_len))
        self.ODA = ODA(d_model, dropout)

    def forward(self, x):
        if self.num_layers:  # Down sample
            x = self.downsampling(x.transpose(1, 2)).permute(0, 2, 1, 3)  # [B L/2**num_layers V D]
        out = self.ODA(x, x, x)  # [B L/2**num_layers V D]
        if self.num_layers:  # Up sample
            out = F.interpolate(out.transpose(1, -1), size=(self.enc_in, self.label_len), mode='bilinear').permute(0, 3, 2, 1)
        return x, out


class ODA(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ODA, self).__init__()
        self.query_projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.key_projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.value_projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.out_projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, V, D = queries.shape
        _, S, _, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries.transpose(1, 3)).permute(0, 2, 3, 1)  # [B V L D]
        keys = self.key_projection(keys.transpose(1, 3)).permute(0, 2, 3, 1)
        values = self.value_projection(values.transpose(1, 3)).permute(0, 2, 3, 1)

        scores = torch.einsum("bvld,bvmd->bvlm", queries, keys)  # [B V L L]
        attn_mask = OffDiagMask(B, L, V, S, device=queries.device)  # [B V L L]
        scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V L L]
        out = torch.einsum("bvls,bvsd->bvld", attn, values)  # [B V L D]

        return self.out_projection(out.transpose(1, 3)).permute(0, 2, 3, 1)  # [B L V D]


class VRCA(nn.Module):
    def __init__(self, dim, enc_in, alpha=0.05, eps=1e-6, dropout=0.1):
        super(VRCA, self).__init__()
        self.vrca = VRCA_Cucconi(dim, enc_in, alpha, eps, dropout)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=dim * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim * 4, out_channels=dim, kernel_size=1)

    def forward(self, x):
        new_x = self.vrca(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x + y)


class VRCA_Cucconi(nn.Module):
    def __init__(self, dim, enc_in, alpha=0.05, eps=1e-6, dropout=0.1):
        super(VRCA_Cucconi, self).__init__()
        self.dim = dim
        self.enc_in = enc_in
        self.alpha = torch.tensor(alpha)
        self.eps = eps
        self.rio = (2 * (4 * self.dim ** 2 - 4)) / ((self.dim * 4 + 1) * (self.dim * 16 + 11)) - 1
        self.sup = ((math.sqrt(-8 * torch.log(self.alpha)) / 2) ** 2) / (1 - self.rio)

        self.queries_projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.keys_projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.value_projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def cucconi(self, queries, keys):
        q = queries.repeat(1, self.enc_in, 1)
        q = q.contiguous().view(q.shape[0], self.enc_in, self.enc_in, self.dim)  # [B V V dim]
        k = keys.repeat(1, self.enc_in, 1)
        k = k.contiguous().view(k.shape[0], self.enc_in, self.enc_in, self.dim)  # [B V V dim]
        k = k.transpose(1, 2)
        value = torch.cat([q, k], dim=-1)
        _, index = torch.sort(value)
        index_q = index[:, :, :, :self.dim] + 1
        square_q = torch.square(index_q)
        sum_square = torch.sum(square_q, dim=-1)  # [B V V]
        rev_square_q = torch.square(self.dim * 2 + 1 - index_q)
        rev_sum_square = torch.sum(rev_square_q, dim=-1)  # [B V V]
        mu = self.dim * (self.dim * 2 + 1) * (self.dim * 4 + 1)
        sigma = math.sqrt(self.dim ** 2 * (self.dim * 2 + 1) * (self.dim * 4 + 1) * (self.dim * 16 + 11) / 5)

        u = (6 * sum_square - mu) / sigma  # [B V V]
        v = (6 * rev_sum_square - mu) / sigma  # [B V V]
        c = (torch.square(u) + torch.square(v) - 2 * self.rio * torch.mul(u, v)) / 2 / (1 - self.rio ** 2)  # [B V V]

        sup = torch.ones_like(c) * self.sup
        mat = torch.gt(c, self.sup)  # c > sup
        out = torch.where(mat, sup, c)  # [B V V]
        return out

    def forward(self, queries, keys, values):
        queries = self.queries_projection(queries.transpose(1, 2)).permute(0, 2, 1)  # [B V dim]
        keys = self.keys_projection(keys.transpose(1, 2)).permute(0, 2, 1)  # [B V dim]
        values = self.value_projection(values.transpose(1, 2)).permute(0, 2, 1)  # [B V dim]

        queries = torch.softmax(queries, dim=-1)  # [B V dim]
        keys = torch.softmax(keys, dim=-1)
        scores = self.cucconi(queries, keys) + self.eps  # [B V V]

        scores = -torch.log(scores)
        scores = self.dropout(torch.softmax(scores, dim=-1))  # [B V V]
        values = self.value_projection(values.transpose(1, 2)).permute(0, 2, 1)  # [B V dim]
        output = torch.einsum("bls,bsd->bld", scores, values)
        output = self.projection(output.transpose(1, 2)).permute(0, 2, 1)  # [B V dim]
        return output
