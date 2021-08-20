#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-20 09:15:53
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from typing import Callable

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split as tts

from text_embeddings.x import X, XLoss


class TweetData(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class LightningX(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 512,
        num_layers: int = 12,
        embed_dim: int = 512,
        num_cross_attention_heads: int = 1,
        num_latent_attention_heads: int = 8,
        num_latent_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        batch_first: bool = False,
        latent_attention: Callable = nn.MultiheadAttention,
    ):
        super().__init__()
        self.model = X(
            num_classes=num_classes,
            latent_dim=latent_dim,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_cross_attention_heads=num_cross_attention_heads,
            num_latent_attention_heads=num_latent_attention_heads,
            num_latent_layers=num_latent_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            batch_first=batch_first,
            latent_attention=latent_attention,
        )
        self.loss = XLoss(nn.CrossEntropyLoss(), 0.8, num_layers)
        self.lr = learning_rate
        self.metric = torchmetrics.F1(2)

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        probas, preds, *_ = self(x, mask)
        loss = self.loss(probas, preds, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        probas, preds, _, pred = self(x, mask)
        loss = self.loss(probas, preds, y)
        _ = self.metric(pred, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_f1 = self.metric.compute()
        self.log(f"val_f1", val_f1, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):

        return {"optimizer": torch.optim.Adagrad(self.parameters(), lr=self.lr)}


def load_datasets(path, test_size: float = 0.3):
    df = pd.read_csv(path)
    assert "target" in df and "text" in df, "Wrong column names in dataset"
    train, val = tts(df, test_size=test_size, stratify=df.target)
    return TweetData(train.to_dict(orient="records")), TweetData(
        val.to_dict(orient="records")
    )


def collate_fn(batch, max_length: int = 1024):

    B = [list(b["text"].encode("utf-8")) for b in batch]
    L = [b["target"] for b in batch]

    ids = np.zeros((len(B), max(max_length, max(map(len, B)))))
    mask = np.zeros((len(B), max(max_length, max(map(len, B)))))

    for i, b in enumerate(B):
        ids[i, : len(b)] = [x + 1 for x in b]  # shift one idx for padding
        mask[i, len(b) :] = 1

    return (
        torch.from_numpy(ids).long(),
        torch.from_numpy(np.asarray(L)).long(),
        torch.from_numpy(mask).bool(),
    )


if __name__ == "__main__":

    embed_dim = 64
    seq_length = 512

    model = LightningX(
        num_classes=2,
        latent_dim=256,
        num_layers=12,
        embed_dim=embed_dim,
        num_cross_attention_heads=1,
        num_latent_attention_heads=2,
        num_latent_layers=2,
        ff_dim=embed_dim * 2,
        dropout=0.1,
        batch_first=True,
        latent_attention=nn.MultiheadAttention,
    )

    train, val = load_datasets(
        "/Users/chenghao/Downloads/nlp-getting-started/train.csv"
    )

    trainer = pl.Trainer(gpus=0, overfit_batches=1)
    trainer.fit(
        model,
        DataLoader(
            train, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4
        ),
        DataLoader(
            val, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4
        ),
    )
