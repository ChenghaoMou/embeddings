#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-27 09:47:26
# @Author       : Chenghao Mou (mouchenghao@gmail.com)


"""This is from paper Charformer: Fast Character Transformers via Gradient-based Subword Tokenization."""
import math


import torch
import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = rearrange(x, "b h l -> l b h")
        pe_ = repeat(
            self.pe,
            "s b h -> (repeat s) b h",
            repeat=math.ceil(x.shape[0] / self.pe.shape[0]),
        )
        x = x + pe_[: x.shape[0], :]
        return rearrange(self.dropout(x), "l b h -> b h l")


class GBST(pl.LightningModule):
    """Gradient-based Subword Tokenization module from the paper:
    Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.

    Parameters
    ----------
    embed_size : int, optional
        The embedding size for each byte/character, by default 259
    max_block_size : int, optional
        Every subword token of length from 1 to max_block_size are considered, by default 4
    downsampling_factor : int, optional
        Downsampling rate from byte sequence to the final sequence, by default 2
    score_calibration : bool, optional
        To calibrate the scores with a self-attention like step, by default True
    vocab_size : int, optional
        The size of the byte vocabulary, by default 256

    Examples
    --------
    >>> model = GBST(
    ...     embed_size=128,
    ...     max_block_size=4,
    ...     downsampling_factor=2,
    ...     score_calibration=True,
    ...     vocab_size=256,
    ... )
    >>> ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))])
    >>> assert ids.shape == torch.Size([1, 33]), ids.shape
    >>> hidden = model(ids)
    >>> assert hidden.shape == torch.Size([1, 17, 128]), hidden.shape
    """

    def __init__(
        self,
        embed_size: int = 256,
        max_block_size: int = 4,
        downsampling_factor: int = 2,
        score_calibration: bool = True,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_block_size = max_block_size
        self.score_calibration = score_calibration
        self.downsampling_factor = downsampling_factor
        self.embed_size = embed_size

        self.byte_embedding = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=0
        )
        self.block_position_embedding = PositionalEncoding(
            self.embed_size, max_len=self.max_block_size
        )

        self.avg_pools = nn.ModuleDict(
            {
                str(i): nn.AvgPool1d(i, ceil_mode=True)
                for i in range(1, self.max_block_size + 1)
            }
        )
        self.block_scorer = nn.Linear(self.embed_size, 1)
        self.down_sampler = nn.AvgPool1d(self.downsampling_factor, ceil_mode=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input):

        byte_embeddings = self.byte_embedding(input)
        sequence_length = byte_embeddings.shape[1]

        Xs = []
        X_scores = []

        for block_size in range(1, self.max_block_size + 1):
            positioned_embeddings = rearrange(byte_embeddings, "b l h -> b h l")
            positioned_embeddings = self.block_position_embedding(positioned_embeddings)

            # b h s
            Xb = self.avg_pools[str(block_size)](positioned_embeddings)
            # b 1 s
            Xb_scores = rearrange(
                self.block_scorer(rearrange(Xb, "b h s -> b s h")), "b s 1 -> b 1 s"
            )
            # b h l
            Xb_ = Xb.repeat_interleave(repeats=block_size, dim=2)
            # b 1 l
            Xb_scores_ = Xb_scores.repeat_interleave(repeats=block_size, dim=2)

            Xs.append(Xb_[:, :, :sequence_length])
            X_scores.append(Xb_scores_[:, :, :sequence_length])

        # b M l
        scores = torch.cat(X_scores, dim=1)
        # b l M 1
        scores = rearrange(torch.softmax(scores, dim=1), "b M l -> b l M 1")

        if self.score_calibration:
            # b l M 1
            scores = (
                torch.softmax(scores @ rearrange(scores, "b l M 1 -> b l 1 M"), dim=-1)
                @ scores
            )

        # b l h M
        Xs = rearrange(torch.stack(Xs, dim=0), "M b h l -> b l h M")
        Xs = rearrange(Xs @ scores, "b l h 1 -> b h l")
        Xs = rearrange(self.down_sampler(Xs), "b h s -> b s h")

        return Xs