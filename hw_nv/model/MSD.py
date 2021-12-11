import torch
import torch.nn as nn


class DS(nn.Module):
    def __init__(self, norm, leaky_slope=0.1):
        super().__init__()
        self.body = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        ])
        self.leaky_slope = leaky_slope

    def forward(self, x):
        # x: [batch_sz, audio_len] raw audio
        out = x
        fmaps = []
        for b in self.body:
            out = b(out)
            out = nn.functional.leaky_relu(out, self.leaky_slope)
            fmaps.append(out)
        return torch.flatten(out, 1, -1), fmaps


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DS(nn.utils.spectral_norm),
            DS(nn.utils.weight_norm),
            DS(nn.utils.weight_norm),
        ])

        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        # x: [batch, 1, audio_len]
        out = []
        fmaps = []
        for i in range(len(self.discriminators)):
            if i != 0:
                x = self.pools[i-1](x)
            cur_out, cur_fmaps = self.discriminators[i](x)
            out.append(cur_out)
            fmaps.append(cur_fmaps)
        return out, fmaps
