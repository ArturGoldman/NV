import torch
import torch.nn as nn


class InBlock(nn.Module):
    def __init__(self, h, kr, dr, leaky_slope=0.1):
        super().__init__()
        self.net = []
        for i in range(len(dr)):
            self.net += [nn.LeakyReLU(leaky_slope),
                         nn.Conv1d(h, h, kr, stride=1, dilation=dr[i], padding='same')]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, h, kr, dr):
        super().__init__()
        self.in_blocks = nn.ModuleList()
        for i in range(len(dr)):
            self.in_blocks.append(InBlock(h, kr, dr[i]))

    def forward(self, x):
        out = x
        for block in self.in_blocks:
            out += block(out)
        return out


class MRF(nn.Module):
    def __init__(self, h, kr, dr):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(kr)):
            self.blocks.append(ResBlock(h, kr[i], dr[i]))

    def forward(self, x):
        out = torch.zeros_like(x)
        for block in self.blocks:
            out += block(x)
        return out


class Generator(nn.Module):
    def __init__(self, h, ku, kr, dr, leaky_slope=0.1):
        """
        h: int, hidden dim
        ku: List[int]
        kr: List[int]
        dr: List[List[List[int]]]
        """
        super().__init__()
        self.start = nn.Conv1d(80, h, 7, 1, padding=3)
        self.body = []
        cur_ch = h
        for i in range(len(ku)):
            self.body += [nn.LeakyReLU(leaky_slope),
                          nn.ConvTranspose1d(cur_ch, cur_ch//2, ku[i], ku[i]//2,  ku[i]//4),
                          MRF(cur_ch//2, kr, dr)]
            cur_ch //= 2
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Sequential(
            nn.LeakyReLU(leaky_slope),
            nn.Conv1d(cur_ch, 1, 7, 1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.start(x)
        out = self.body(out)
        out = self.tail(out)
        return out