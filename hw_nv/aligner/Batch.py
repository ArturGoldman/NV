import torch
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    melspec: torch.Tensor
    melspec_length: torch.Tensor
    durations: Optional[torch.Tensor] = None
    pad_value: float = -11.5129251

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.melspec = self.melspec.to(device)
        self.melspec_length = self.melspec_length.to(device)
        if self.durations is not None:
            self.durations = self.durations.to(device)
