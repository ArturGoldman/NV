import torch
from torch import nn, Tensor
from hw_tts.aligner import Batch
from hw_tts.processing import MelSpectrogram


class FTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.melspec = MelSpectrogram(config)
        # self.loss = nn.MSELoss()

    def __call__(self, outputs: Tensor, batch: Batch):
        # outputs: [batch_sz, seq_len, n_mels]

        MSE = 0
        for i in range(outputs.size(0)):
            gts = self.melspec(batch.waveform[i, :batch.waveform_length[i]])
            MSE += ((outputs[i, :batch.alignment[i].sum()] - gts) ** 2).mean()

        return MSE / outputs.size(0)


class DPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1/2

    def __call__(self, batch: Batch, pred: Tensor):
        # pred: [batch_sz, lens], lens are not compatible with gt
        MSE = 0
        for i in range(pred.size(0)):
            gt = torch.from_numpy(batch.alignment[i]).float()
            gt[gt == 0] = self.eps
            MSE += ((pred[i, :batch.token_lengths[i]].cpu() - torch.log(gt)) ** 2).mean()
        return MSE / pred.size(0)
