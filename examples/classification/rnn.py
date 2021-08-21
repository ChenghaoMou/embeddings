#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-05-22 14:50:34
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional
from text_embeddings.visual import VTRTokenizer
from einops import rearrange


class Model(pl.LightningModule):
    def __init__(
        self, hidden: int = 128, learning_rate: float = 1e-3, num_labels: int = 20
    ):
        super().__init__()
        self.model = nn.GRU(
            hidden, hidden, num_layers=2, bidirectional=True, batch_first=True
        )
        self.nonlinear = nn.ReLU()
        self.fc = nn.Linear(hidden * 2, num_labels)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.lr = learning_rate

    def forward(self, batch):

        embeddings = batch["input_ids"].float()
        logits, _ = self.model(rearrange(embeddings, "b s h w -> b s (h w)"))
        logits = torch.cat(
            [
                logits[:, :, : logits.shape[-1] // 2],
                logits[:, :, logits.shape[-1] // 2 :],
            ],
            dim=-1,
        )
        logits = torch.mean(logits, dim=1)
        logits = self.nonlinear(logits)
        logits = self.fc(logits)

        return logits

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        logits = self.forward(inputs)
        return {"loss": self.loss(logits, labels)}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        # logger.debug(f"{labels.shape, logits.shape}")
        loss = self.loss(logits, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):

        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.lr)}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        font_path="/home/chenghaomou/embeddings/Noto_Sans/NotoSans-Regular.ttf",
        font_size: int = 16,
        window_size: int = 8,
        stride: int = 5,
        batch_size: int = 8,
        subtask: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = (
            load_dataset(dataset_name, subtask)
            if subtask
            else load_dataset(dataset_name)
        )
        self.tokenizer = VTRTokenizer(
            font=font_path, window_size=window_size, font_size=font_size, stride=stride
        )
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = self.dataset["train"]
        self.val = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            [{"text": x["text"], "label": x["label"]} for x in self.train],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            [{"text": x["text"], "label": x["label"]} for x in self.val],
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def collate_fn(self, examples):

        text = [e["text"] for e in examples]
        labels = [e["label"] for e in examples]

        results = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return results, torch.from_numpy(np.asarray(labels)).long()


if __name__ == "__main__":

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(Model, DataModule)
