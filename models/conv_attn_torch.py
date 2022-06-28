from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchpack.utils.config import configs


__all__ = ['ConvAttnTorchTSRRHR',
            'ConvAttnTorchTSRRHR_v']


class ConvAttnTorchTSRRHR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, n_head,
                 conv_layer_num, attn_layer_num):
        super().__init__()

        self.input_layer_t = nn.Conv1d(
            in_channels=2*in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.layers_t = nn.ModuleList()
        for i in range(conv_layer_num):
            self.layers_t.append(nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))

        self.input_layer_s = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.layers_s = nn.ModuleList()
        for i in range(conv_layer_num):
            self.layers_s.append(nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))

        self.fusion_layer = nn.Conv1d(
            in_channels=2*out_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_ch,
                                                   nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=attn_layer_num)

        self.regress_rr = nn.Linear(out_ch, 1)
        self.regress_hr = nn.Linear(out_ch, 1)

    def forward(self, x):

        if 'spec_seq' in configs.model.feats:
            x_spec = x['spec_seq']
        if 'tmpo_seq' in configs.model.feats:
            x_tmpo = x['tmpo_seq']

        x_spec = x_spec.permute(0, 2, 1)  # N, C, L
        x_spec = self.input_layer_s(x_spec)

        for i in range(len(self.layers_s)):
            x_spec = F.relu(self.layers_s[i](x_spec))

        x_tmpo = x_tmpo.permute(0, 2, 1)  # N, C, L
        x_tmpo = self.input_layer_t(x_tmpo)

        for i in range(len(self.layers_t)):
            x_tmpo = F.relu(self.layers_t[i](x_tmpo))

        x = torch.cat([x_spec, x_tmpo], dim=1)
        x = self.fusion_layer(x)

        x = x.permute(2, 0, 1)  # L, N, C
        x = self.encoder(x)
        x = x.permute(1, 2, 0)  # N, C, L

        x = F.avg_pool1d(x, kernel_size=configs.dataset.resample_len).squeeze()
        x_r = self.regress_rr(x)
        x_h = self.regress_hr(x)

        return x_r.squeeze(dim=-1), x_h.squeeze(dim=-1)


class ConvAttnTorchTSRRHR_v(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, n_head,
                 conv_layer_num, attn_layer_num):
        super().__init__()

        self.input_layer_t = nn.Conv1d(
            in_channels=2*in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.layers_t = nn.ModuleList()
        for i in range(conv_layer_num):
            self.layers_t.append(nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))

        self.input_layer_s = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        self.layers_s = nn.ModuleList()
        for i in range(conv_layer_num):
            self.layers_s.append(nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))

        self.fusion_layer = nn.Conv1d(
            in_channels=2*out_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_ch,
                                                   nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=attn_layer_num)

        self.regress_rr = nn.Linear(configs.dataset.resample_len, 1)
        self.regress_hr = nn.Linear(configs.dataset.resample_len, 1)

    def forward(self, x):

        if 'spec_seq' in configs.model.feats:
            x_spec = x['spec_seq']
        if 'tmpo_seq' in configs.model.feats:
            x_tmpo = x['tmpo_seq']

        x_spec = x_spec.permute(0, 2, 1)  # N, C, L
        x_spec = self.input_layer_s(x_spec)

        for i in range(len(self.layers_s)):
            x_spec = F.relu(self.layers_s[i](x_spec))

        x_tmpo = x_tmpo.permute(0, 2, 1)  # N, C, L
        x_tmpo = self.input_layer_t(x_tmpo)

        for i in range(len(self.layers_t)):
            x_tmpo = F.relu(self.layers_t[i](x_tmpo))

        x = torch.cat([x_spec, x_tmpo], dim=1)
        x = self.fusion_layer(x)

        x = x.permute(2, 0, 1)  # L, N, C
        x = self.encoder(x)
        x = x.permute(1, 2, 0)  # N, C, L

        x = x.permute(0, 2, 1)  # N, C, L

        x = F.avg_pool1d(x, kernel_size=configs.model.out_ch).squeeze()

        x_r = self.regress_rr(x)
        x_h = self.regress_hr(x)

        return x_r.squeeze(dim=-1), x_h.squeeze(dim=-1)


