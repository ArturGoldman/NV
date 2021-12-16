import torch
import librosa
import torch.nn as nn
import torchaudio


class MelSpectrogram(nn.Module):
    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        self.config = config
        melspec_conf = self.config["MelSpectrogram"]

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=melspec_conf["sr"],
            win_length=melspec_conf["win_length"],
            hop_length=melspec_conf["hop_length"],
            n_fft=melspec_conf["n_fft"],
            f_min=melspec_conf["f_min"],
            f_max=melspec_conf["f_max"],
            n_mels=melspec_conf["n_mels"],
            center=False
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = melspec_conf["power"]

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=melspec_conf["sr"],
            n_fft=melspec_conf["n_fft"],
            n_mels=melspec_conf["n_mels"],
            fmin=melspec_conf["f_min"],
            fmax=melspec_conf["f_max"]
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
        self.hop_size=melspec_conf["hop_length"]
        self.n_fft=melspec_conf["n_fft"]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T]
        """
        audio = torch.nn.functional.pad(audio, (int((self.n_fft-self.hop_size)/2), int((self.n_fft-self.hop_size)/2)), mode='reflect')
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel