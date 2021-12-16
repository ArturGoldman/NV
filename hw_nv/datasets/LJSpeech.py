import torch
import torchaudio
import random
import numpy as np
from hw_nv.processing import MelSpectrogram


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, config, root, segment_size=None, to_sr=22050, limit=None):
        super().__init__(root=root)
        cur_sz = super().__len__()
        self._index = list(range(cur_sz))
        self.limit = limit
        self.to_sr = to_sr
        self.sz = segment_size
        self.melspectr = MelSpectrogram(config)
        random.seed(42)
        random.shuffle(self._index)
        if limit is not None:
            self._index = self._index[:limit]

    def __getitem__(self, index: int):
        waveform, old_sr, _, transcript = super().__getitem__(self._index[index])
        waveform = torchaudio.transforms.Resample(old_sr, self.to_sr)(waveform)
        if self.sz is not None:
            rand_pos = random.randint(0, waveform.size(-1) - self.sz)
            waveform = waveform[:, rand_pos:rand_pos+self.sz]
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        melspec = self.melspectr(waveform)
        melspec_length = torch.tensor([waveform.shape[0]]).int()

        return waveform, waveform_length, transcript, melspec, melspec_length

    def __len__(self):
        return len(self._index)
