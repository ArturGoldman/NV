import argparse
from pathlib import Path
import shutil

import torch
from tqdm import tqdm

import torchaudio

import hw_tts.model as module_arch
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.datasets import english_cleaners
from hw_tts.model import Vocoder
from hw_tts.aligner import Batch

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config):
    logger = config.get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)
    logger.info(sum(p.numel() for p in model.parameters()))

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    save_path = Path('./audios')
    if save_path.exists():
        shutil.rmtree(str(save_path), ignore_errors=True)
    save_path.mkdir(parents=True, exist_ok=False)

    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    vocoder = Vocoder().to(device)

    f = open(config["file_dir"], 'r')
    for i, line in tqdm(enumerate(f.readlines())):
        transcript = english_cleaners(line)
        tokens, token_lengths = tokenizer(transcript)
        batch = Batch(None, None, None, tokens.to(device), token_lengths.to(device), None)
        spec_pred = model(batch, device)
        pred_wav = vocoder.inference(spec_pred.transpose(-1, -2)).cpu()
        torchaudio.save(str(save_path)+'/'+str(i+1)+'.wav', pred_wav, sample_rate=22050)

    print("Testing: DONE")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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

    config = ConfigParser.from_args(args)

    main(config)