import random
from random import shuffle

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import PIL
from torchvision.transforms import ToTensor

import io
import matplotlib.pyplot as plt

from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf, plot_attention_to_buf
from hw_nv.utils import inf_loop, MetricTracker
from hw_nv.aligner import Batch
from hw_nv.processing import MelSpectrogram


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            Generator,
            MPD,
            MSD,
            optimizer_g,
            optimiser_d,
            config,
            device,
            data_loader,
            val_data_loader=None,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(Generator, MSD, MPD, optimizer_g, optimiser_d, lr_scheduler_g, lr_scheduler_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.attentions = None
        self.melspec = MelSpectrogram(config).to(device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.log_step = self.config["trainer"]["log_step"]

        self.train_metrics = MetricTracker(
            "total_disc_loss", "total_gen_loss",
            "grad norm g", "grad norm mpd", "grad norm msd",
            writer=self.writer
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.msd.train()
        self.mpd.train()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            to_log = False
            if batch_idx + 1 == self.len_epoch or batch_idx + 1 == self.len_epoch // 2:
                to_log = True
            try:
                l_disc, l_gen = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    to_log=to_log
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.msd.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.mpd.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm g", self.get_grad_norm(self.generator))
            # self.train_metrics.update("grad norm msd", self.get_grad_norm(self.msd))
            # self.train_metrics.update("grad norm mpd", self.get_grad_norm(self.msd))
            if batch_idx % self.log_step == 0 or (batch_idx + 1 == self.len_epoch and epoch == self.epochs):
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss_disc: {:.6f} Loss_gen: {:.6f}".format(
                        epoch, self._progress(batch_idx), l_disc, l_gen
                    )
                )

                self._log_scalars(self.train_metrics)
            if batch_idx + 1 >= self.len_epoch:
                break

        self.lr_scheduler_g.step()
        self.lr_scheduler_d.step()
        self.writer.add_scalar("learning rate", self.lr_scheduler_g.get_last_lr()[0])
        self._valid_example()

        log = self.train_metrics.result()

        return log

    @staticmethod
    def discriminator_loss(gt: torch.Tensor, pred: torch.Tensor):
        loss = 0
        for i in range(len(gt)):
            true_p = torch.mean((gt[i] - 1) ** 2)
            false_p = torch.mean(pred[i] ** 2)
            loss += true_p + false_p
        return loss

    def mel_loss(self, pred, batch):
        pred_mel = self.melspec(pred).squeeze()

        return torch.nn.functional.l1_loss(pred_mel, batch.melspec)

    @staticmethod
    def generator_loss(pred):
        loss = 0
        for i in range(len(pred)):
            false_p = torch.mean(pred[i] ** 2)
            loss += false_p
        return loss

    @staticmethod
    def fmaps_loss(gt, pred):
        loss = 0
        for a, b in zip(gt, pred):
            for c, d in zip(a, b):
                loss += torch.mean(torch.abs(c - d))
        return loss

    def process_batch(self, batch: Batch, metrics: MetricTracker, to_log: bool):
        batch.to(self.device)
        outputs = self.generator(batch.melspec)
        """
        self.optimizer_d.zero_grad()


        msd_true, _ = self.msd(batch.waveform)
        msd_false, _ = self.msd(outputs.detach())
        loss_msd = self.discriminator_loss(msd_true, msd_false)

        mpd_true, _ = self.mpd(batch.waveform)
        mpd_false, _ = self.mpd(outputs.detach())
        loss_mpd = self.discriminator_loss(mpd_true, mpd_false)

        total_disc_loss = loss_msd + loss_mpd
        total_disc_loss.backward()
        self.optimizer_d.step()
        """
        total_disc_loss = torch.tensor(0)
        self.optimizer_g.zero_grad()
        loss_mel = self.mel_loss(outputs, batch)
        """
        msd_true, msd_true_fmaps = self.msd(batch.waveform)
        msd_false, msd_false_fmaps = self.msd(outputs)
        loss_msd_fmaps = self.fmaps_loss(msd_true_fmaps, msd_false_fmaps)

        mpd_true, mpd_true_fmaps = self.mpd(batch.waveform)
        mpd_false, mpd_false_fmaps = self.mpd(outputs)
        loss_mpd_fmaps = self.fmaps_loss(mpd_true_fmaps, mpd_false_fmaps)

        loss_gen_msd = self.generator_loss(msd_false)
        loss_gen_mpd = self.generator_loss(mpd_false)
        """

        # total_gen_loss = 45*loss_mel + 2*(loss_msd_fmaps+loss_mpd_fmaps)+loss_gen_msd+loss_gen_mpd
        total_gen_loss = loss_mel
        total_gen_loss.backward()
        self.optimizer_g.step()

        if to_log:
            j = random.randint(0, outputs.size(0) - 1)
            pred_melspec = self.melspec(outputs[j].detach()).squeeze()
            self._log_spectrogram("train_pred_mel", pred_melspec)
            self._log_spectrogram("train_ground_truth_mel", batch.melspec[j])
            self._log_audios("train_pred", outputs[j].detach())
            self._log_audios("train_ground_truth", batch.waveform[j])

        metrics.update("total_disc_loss", total_disc_loss.item())
        metrics.update("total_gen_loss", total_gen_loss.item())

        return total_disc_loss.item(), total_gen_loss.item()

    def _valid_example(self, n_examples=1):
        """
        see how model works on example
        """
        self.generator.eval()
        self.msd.eval()
        self.mpd.eval()

        with torch.no_grad():
            for i in range(n_examples):
                batch = next(iter(self.val_data_loader))
                batch.to(self.device)
                outputs = self.generator(batch.melspec)
                pred_melspec = self.melspec(outputs[0])
                self._log_spectrogram("val_pred_mel", pred_melspec.squeeze())
                self._log_spectrogram("val_ground_truth_mel", batch.melspec[0])
                self._log_audios("val_pred", outputs[0].detach())
                self._log_audios("val_ground_truth", batch.waveform[0])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, name, spec):
        image = PIL.Image.open(plot_spectrogram_to_buf(spec.cpu()))
        self.writer.add_image(name, (ToTensor()(image)).transpose(-1, -2).flip(-2))

    def _log_audios(self, name, audio_example):
        audio = audio_example
        self.writer.add_audio(name, audio, sample_rate=self.config["MelSpectrogram"]["sr"])

    def _log_attention(self, buf):
        image = PIL.Image.open(buf).rotate(270, expand=True)
        self.writer.add_image("Attentions", (ToTensor()(image)).transpose(-1, -2).flip(-2))

    @staticmethod
    @torch.no_grad()
    def get_grad_norm(model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
