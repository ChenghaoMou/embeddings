#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-05-06 21:13:40
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Twitter sentiment classification with visual text representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split as tts
from typing import *
from transformers.tokenization_utils_base import *
from einops import rearrange

class Classifier(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.conv = nn.Conv2d(1, 1, 5)
        self.rnn = nn.GRU(self.hparams["hidden_size"], self.hparams["hidden_size"], bidirectional=True)
        self.linear = nn.Linear(self.hparams["hidden_size"] * 2, self.hparams["num_class"])
        self.loss = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, batch):
        inputs = batch["input_ids"].float()
        batch_size = inputs.shape[0]
        inputs = rearrange(inputs, "B S H W -> (B S) 1 H W")
        hidden = self.conv(inputs)
        hidden = rearrange(inputs, "(B S) 1 H W -> B S (H W)", B = batch_size)
        hidden, _ = self.rnn(F.relu(hidden))
        hidden = torch.cat([hidden[:, :, :hidden.shape[-1]//2], hidden[:, :, hidden.shape[-1]//2:]], dim=-1)
        hidden, _ = torch.max(hidden, dim=1) # max over the sequence
        logits = self.linear(F.relu(hidden))

        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)
        return {
            "loss": self.loss(logits.reshape(-1, logits.shape[-1]), batch["labels"].view(-1))
        }
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), batch["labels"].view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return {
            "val_loss": loss
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }
    
class DummyDataset(Dataset):

    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class DummyDataModule(pl.LightningDataModule):

    def __init__(self, path, tokenizer, label2idx, batch_size, test_size: float = 0.2, random_state: int = 42):
        super().__init__()
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.data = pd.read_csv(path)
        self.train, self.val = tts(self.data, test_size=test_size, random_state=random_state, stratify=self.data.airline_sentiment)
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self) -> Any:
        return DataLoader(DummyDataset(self.train.to_dict("records")), batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)
    
    def train_dataloader(self) -> Any:
        return DataLoader(DummyDataset(self.train.to_dict("records")), batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def collate_fn(self, batch):
        text = [r["text"] for r in batch]
        labels = [r["airline_sentiment"] for r in batch]
        result = self.tokenizer(text, return_tensors="pt", padding=PaddingStrategy.LONGEST, truncation=TruncationStrategy.LONGEST_FIRST)
        result["labels"] = torch.LongTensor([self.label2idx[l] for l in labels])
        return result

if __name__ == "__main__":

    from text_embeddings.visual import VTRTokenizer

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        overfit_batches=1,
    )

    model = Classifier({
        "lr": 1e-1,
        "hidden_size": 14 * 10,
        "num_class": 3,
    })
    tokenizer = VTRTokenizer(
        font_size=14,
        window_size=10,
        font="~/Library/Fonts/NotoSansDisplay-Regular.ttf",
        max_length=36
    )

    data = DummyDataModule("/Users/chenghaomou/Downloads/archive/Tweets.csv", tokenizer, {"negative": 0, "neutral": 1, "positive": 2}, batch_size=8)
    trainer.fit(model, data)
