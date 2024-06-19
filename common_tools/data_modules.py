from typing import Dict, List, Optional, Union
from torch.utils.data.dataloader import DataLoader
import sys

sys.path.append("../..")
import common_tools.utils as myut
import pytorch_lightning as pl


class MicroDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data,
            train_transform,
            train_batch_size: int,
            val_data,
            val_transform,
            val_batch_size: int,
            num_workers: int,
    ):
        super().__init__()
        self.train_data = train_data
        self.train_transform = train_transform
        self.train_batch_size = train_batch_size
        self.val_data = val_data
        self.val_transform = val_transform
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str]) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = myut.get_dataset(self.train_data, self.train_transform)
            self.val_dataset = myut.get_dataset(self.val_data, self.val_transform)
            self.train_loader = myut.get_dataloader(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                drop_last=False,
            )
            self.val_loader = myut.get_dataloader(
                dataset=self.val_dataset,
                shuffle=False,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                drop_last=False,
            )

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return self.train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_loader
