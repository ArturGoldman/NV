from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import hw_nv.collate_fn
from hw_nv.datasets import LJSpeechDataset
from hw_nv.collate_fn import LJSpeechCollator
from hw_nv.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        drop_last = False
        if split == "train":
            drop_last = True

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(LJSpeechDataset(configs, **ds["args"]))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        collate_fn = configs.init_obj(params["collator"], hw_nv.collate_fn)
        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
