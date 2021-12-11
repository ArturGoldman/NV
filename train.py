import argparse
import collections
import warnings
import itertools

import numpy as np
import torch

import hw_nv.model as module_arch
from hw_nv.model import MSD, MPD
from hw_nv.datasets.utils import get_dataloaders
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    # build model architecture, then print to console
    model_g = config.init_obj(config["arch"], module_arch)
    logger.info(model_g)
    logger.info(sum(p.numel() for p in model_g.parameters()))


    model_msd = MSD()
    logger.info(model_msd)
    logger.info(sum(p.numel() for p in model_msd.parameters()))

    model_mpd = MPD()
    logger.info(model_mpd)
    logger.info(sum(p.numel() for p in model_mpd.parameters()))
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model_g = model_g.to(device)
    model_msd = model_msd.to(device)
    model_mpd = model_mpd.to(device)


    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_g = filter(lambda p: p.requires_grad, model_g.parameters())
    optimizer_g = config.init_obj(config["optimizer"], torch.optim, trainable_params_g)
    lr_scheduler_g = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_g)

    trainable_params_msd = filter(lambda p: p.requires_grad, model_msd.parameters())
    trainable_params_mpd = filter(lambda p: p.requires_grad, model_mpd.parameters())
    optimizer_d = config.init_obj(config["optimizer"], torch.optim, itertools.chain(trainable_params_msd,
                                                                                    trainable_params_mpd))
    lr_scheduler_d = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_d)
    trainer = Trainer(
        model_g,
        model_msd,
        model_mpd,
        optimizer_g,
        optimizer_d,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        val_data_loader=dataloaders["val"],
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
