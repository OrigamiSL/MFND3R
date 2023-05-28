# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.RevIN import RevIN
from MFND3R.Modules import *
from MFND3R.embed import DataEmbedding


class R_stage(nn.Module):
    def __init__(self, enc_in, c_out, input_len, ODA_layers=3, VRCA_layers=1, d_model=64, representation=320
                 , dropout=0.0, alpha=0.05, use_RevIN=True):
        super(R_stage, self).__init__()
        self.enc_in = enc_in
        self.input_len = input_len
        self.c_out = c_out
        self.d_model = d_model
        self.use_RevIN = use_RevIN
        self.ODA_layers = ODA_layers
        self.VRCA_layers = VRCA_layers
        if use_RevIN:
            self.revin = RevIN(enc_in)
        self.Embed = DataEmbedding(d_model, dropout, position=True)
        self.HODA = HODA(ODA_layers, input_len, enc_in, d_model, dropout=dropout)

        if VRCA_layers:
            VRCA_layer = [VRCA(representation, enc_in, alpha=alpha, dropout=dropout)
                          for _ in range(VRCA_layers)]
            self.VRCA = nn.ModuleList(VRCA_layer)
            self.flatten = nn.Flatten(2)
            self.linear = nn.Sequential(
                nn.Linear(input_len * d_model, input_len),
                nn.Linear(input_len, input_len),
                nn.Linear(input_len, representation))
            self.projection = nn.Sequential(nn.Linear(representation, representation),
                                            nn.Linear(representation, input_len))
        else:
            self.linear = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, 4 * d_model),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, 1))

    def forward(self, x_enc):
        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')
        x_enc = x_enc.unsqueeze(-1)

        H_enc = self.Embed(x_enc)  # [B L V D]
        H_tem = self.HODA(H_enc)  # [B L V D]

        if self.VRCA_layers:
            recon = self.flatten(H_tem.transpose(1, 2))  # [B V LD]
            recon = self.linear(recon)  # [B V repr]
            for vrca in self.VRCA:
                H_recon = vrca(recon)  # [B V repr]
                recon = H_recon
            recon = self.projection(H_recon)  # [B V L]
            recon = recon.transpose(1, 2)  # [B L V]
        else:
            recon = self.linear(H_tem)  # [B L V 1]
            recon = recon.reshape(-1, recon.shape[1], recon.shape[2])

        x_index = torch.arange(0, 2 ** (self.ODA_layers - 1)).to(recon.device).reshape(-1)
        sigma = 2 ** (self.ODA_layers - 1) / 3
        gauss = torch.exp(-(x_index - x_index[-1]) ** 2 / (2 * sigma ** 2))
        gauss = torch.cat([torch.zeros(recon.shape[1] - 2 ** (self.ODA_layers - 1)).to(recon.device), gauss], dim=0)
        gauss = gauss.unsqueeze(0).unsqueeze(-1).repeat(recon.shape[0], 1, recon.shape[-1])
        recon = (1 - gauss) * recon + gauss * x_enc.squeeze(-1)

        if self.use_RevIN:
            output = self.revin(recon, 'denorm')
        else:
            output = recon

        return output, recon


class F_stage(nn.Module):
    def __init__(self, enc_in, input_len, pred_len, d_model, use_RevIN):
        super(F_stage, self).__init__()
        self.pred_len = pred_len
        self.input_len = input_len
        self.enc_in = enc_in
        self.d_model = d_model

        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=input_len, out_channels=max(4 * d_model, 2 * pred_len), kernel_size=1, padding=0),
            nn.Conv1d(in_channels=max(4 * d_model, 2 * pred_len), out_channels=pred_len, kernel_size=1, padding=0))
        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.revin = RevIN(enc_in)

    def forward(self, recon, x_enc):
        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')  # [B L V]
        output = self.linear(recon)  # [B L_predV]

        if self.use_RevIN:
            output = self.revin(output, 'denorm')
        return output


class MFND3R(nn.Module):
    def __init__(self, enc_in, c_out, input_len, pred_len, ODA_layers=3, VRCA_layers=1, d_model=64, representation=320,
                 dropout=0.0, alpha=0.05, use_RevIN=True):
        super(MFND3R, self).__init__()
        self.input_len = input_len
        self.ODA_layers = ODA_layers
        self.R_models = R_stage(enc_in, c_out, input_len, ODA_layers, VRCA_layers, d_model,
                                representation, dropout, alpha, use_RevIN)
        self.F_model = F_stage(enc_in, input_len, pred_len, d_model, use_RevIN)

    def forward(self, x):
        r_output, recon = self.R_models(x)
        f_output = self.F_model(recon, x)
        return r_output, f_output, x
