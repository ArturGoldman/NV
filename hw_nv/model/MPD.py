import torch
import torch.nn as nn


class DP(nn.Module):
    def __init__(self, n, leaky_slope=0.1):
        super().__init__()
        self.n = n
        self.body = []
        prev_ch = 1
        cur_ch = 32
        for i in range(4):
            self.body += [
                nn.utils.weight_norm(nn.Conv2d(prev_ch, cur_ch, (5, 1), (3, 1), padding=(2, 0)))
            ]
            prev_ch = cur_ch
            cur_ch *= 4
        self.body = nn.ModuleList(self.body)
        self.tail = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(prev_ch, 1024, (5, 1), 1, padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        ])
        self.leaky_slope = leaky_slope

    def forward(self, x):
        # x: [batch_sz, 1, audio_len] raw audio
        b, t = x.size()
        if t % self.n != 0:
            to_pad = self.n - (t % self.n)
            x = nn.functional.pad(x, (0, to_pad), "reflect")
            t += to_pad

        x = x.view(b, 1, t//self.n, self.n)
        fmaps = []
        for block in self.body:
            x = block(x)
            x = nn.functional.leaky_relu(x, self.leaky_slope)
            fmaps.append(x)
        for block in self.tail:
            x = block(x)
            x = nn.functional.leaky_relu(x, self.leaky_slope)
            fmaps.append(x)
        return torch.flatten(x, 1, -1), fmaps


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DP(2),
            DP(3),
            DP(5),
            DP(7),
            DP(11)
        ])

    def forward(self, x):
        out = []
        fmaps = []
        for D in self.discriminators:
            cur_out, cur_fmaps = D(x)
            out.append(cur_out)
            fmaps.append(cur_fmaps)
        return out, fmaps