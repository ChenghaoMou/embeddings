# -*- coding: utf-8 -*-
"""X.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yYthxWqOxtTAcCVqFkEq9UagQxnsC6bf
"""

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
        _ = self.metric(torch.argmax(pred, dim=1), y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_f1 = self.metric.compute()
        self.log(f"val_f1", val_f1, prog_bar=True, on_epoch=True)
        self.log(
            f"val_loss",
            torch.stack([o["val_loss"] for o in outputs]).mean(),
            prog_bar=True,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min")
        return {"optimizer": optim, "scheduler": scheduler}


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

    train, val = load_datasets("train.csv")

    train_dataloader = DataLoader(
        train, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_dataloader = (
        DataLoader(
            val, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2
        ),
    )

    embed_dim = 256

    model = LightningX(
        num_classes=2,
        latent_dim=256,
        num_layers=18,
        embed_dim=embed_dim,
        num_cross_attention_heads=1,
        num_latent_attention_heads=8,
        num_latent_layers=1,
        ff_dim=embed_dim * 2,
        dropout=0.1,
        batch_first=True,
        latent_attention=nn.MultiheadAttention,
        learning_rate=5e-3,
    )

    trainer = pl.Trainer(
        gpus=2,
        val_check_interval=0.25,
        accumulate_grad_batches=4,
        accelerator="ddp",
        plugins="ddp_sharded",
    )
    trainer.fit(model, train_dataloader, val_dataloader)
