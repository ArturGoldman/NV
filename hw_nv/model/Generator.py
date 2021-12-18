import torch
import torch.nn as nn


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class InBlock(nn.Module):
    def __init__(self, h, kr, dr, leaky_slope=0.1):
        super().__init__()
        self.net = []
        for i in range(len(dr)):
            self.net += [nn.LeakyReLU(leaky_slope),
                         nn.Conv1d(h, h, kr, stride=1, dilation=dr[i], padding='same')]
        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights)

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
            out = out + block(out)
        return out


class MRF(nn.Module):
    def __init__(self, h, kr, dr):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.kr = kr
        for i in range(len(kr)):
            self.blocks.append(ResBlock(h, kr[i], dr[i]))

    def forward(self, x):
        out = None
        for block in self.blocks:
            if out is None:
                out = block(torch.clone(x))
            else:
                out = out + block(torch.clone(x))
        return out / len(self.kr)


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
                          nn.ConvTranspose1d(cur_ch, cur_ch // 2, ku[i], ku[i] // 2, (ku[i] - ku[i] // 2) // 2),
                          MRF(cur_ch // 2, kr, dr)]
            cur_ch //= 2
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Sequential(
            nn.LeakyReLU(leaky_slope),
            nn.Conv1d(cur_ch, 1, 7, 1, padding=3),
            nn.Tanh()
        )

        self.body.apply(init_weights)
        self.tail.apply(init_weights)

    def forward(self, x):
        out = self.start(x)
        out = self.body(out)
        out = self.tail(out)
        return out