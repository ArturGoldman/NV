from typing import Tuple, Dict, Optional, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from hw_nv.aligner import Batch


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript, melspec, melspec_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1).unsqueeze(1)
        waveform_length = torch.cat(waveform_length)

        melspec = pad_sequence([
            melspec_[0].transpose(-1, -2) for melspec_ in melspec
        ], batch_first=True, padding_value=Batch.pad_value).transpose(-1, -2)
        melspec_length = torch.cat(melspec_lengths)

        return Batch(waveform, waveform_length, transcript, melspec, melspec_length)