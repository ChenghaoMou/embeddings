#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-08-19 12:47:53
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

"""X is a Perceiver-based encoder model that incorporates byte hash embeddings, learned token pruning and layer wise adaptive computation (inspired from PonderNet)."""

import math
from typing import Callable

import torch
import torch.nn as nn

from torch import Tensor
from einops import repeat, rearrange
from transformers import CanineModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            x = x.transpose(1, 0)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) if not self.batch_first else self.dropout(x).transpose(0, 1)


class AttentionWrapper(nn.Module):
    def __init__(
        self,
        attention_class: Callable,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        batch_first: bool,
        is_cross_attention: bool,
    ):
        super().__init__()

        self.is_cross_attention = is_cross_attention
        self.pre_attention_q_norm = nn.LayerNorm(embed_dim)
        self.pre_attention_kv_norm = (
            nn.LayerNorm(embed_dim) if is_cross_attention else None
        )

        self.attention = attention_class(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        mask: Tensor = None,
    ):
        query = self.pre_attention_q_norm(query)
        key = (
            self.pre_attention_kv_norm(key)
            if key is not None and self.pre_attention_kv_norm is not None
            else key
        )
        value = (
            self.pre_attention_kv_norm(value)
            if value is not None and self.pre_attention_kv_norm is not None
            else value
        )

        # mask is only useful for cross attention, ignore attention weights
        attn_output, *_ = (
            self.attention(query, key, value, key_padding_mask=mask)
            if self.is_cross_attention
            else self.attention(query, query, query)
        )
        output = attn_output + query
        output = self.ff(output) + output

        return output


class XLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_cross_attention_heads: int,
        num_latent_attention_heads: int,
        num_latent_layers: int,
        ff_dim: int,
        dropout: float,
        batch_first: bool,
        latent_attention: Callable,
    ):
        super().__init__()

        self.cross_attention = AttentionWrapper(
            attention_class=nn.MultiheadAttention,
            embed_dim=embed_dim,
            num_heads=num_cross_attention_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            batch_first=batch_first,
            is_cross_attention=True,
        )

        # pesudo transfomer
        self.latent_attentions = nn.ModuleList(
            [
                AttentionWrapper(
                    attention_class=latent_attention,
                    embed_dim=embed_dim,
                    num_heads=num_latent_attention_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    batch_first=batch_first,
                    is_cross_attention=False,
                )
                for _ in range(num_latent_layers)
            ]
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        mask: Tensor = None,
    ):
        o = self.cross_attention(
            query,
            key,
            value,
            mask=mask,
        )

        for attn in self.latent_attentions:
            o = attn(o)

        return o


class X(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        num_layers: int,
        embed_dim: int,
        num_cross_attention_heads: int,
        num_latent_attention_heads: int,
        num_latent_layers: int,
        ff_dim: int,
        dropout: float,
        batch_first: bool,
        max_length: int,
        latent_attention: Callable,
    ):
        super().__init__()

        self.embedding = CanineModel.from_pretrained('google/canine-s')
        self.embedding_ff = nn.Linear(768, embed_dim)
        self.layers = nn.ModuleList(
            [
                XLayer(
                    embed_dim=embed_dim,
                    num_cross_attention_heads=num_cross_attention_heads,
                    num_latent_attention_heads=num_latent_attention_heads,
                    num_latent_layers=num_latent_layers,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    batch_first=batch_first,
                    latent_attention=latent_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_classes = num_classes
        self.latent = nn.Parameter(torch.rand((latent_dim, embed_dim)))
        self.output_layer = nn.Linear(embed_dim, self.num_classes)
        self.lambda_layer = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(
        self,
        **inputs
    ):
        
        mask = inputs.get("attention_mask", None)
        with torch.no_grad():
            outputs = self.embedding(**inputs)
            x = outputs.last_hidden_state
        
        x = self.embedding_ff(x)
        batch_size, *_ = x.shape
        un_halted_prob = x.new_ones((batch_size,))
        halted = x.new_zeros((batch_size,))

        latent = repeat(
            rearrange(self.latent, "N D -> 1 N D"), "1 N D -> B N D", B=batch_size
        )

        probas = []
        preds = []

        p_m = x.new_zeros((batch_size,))
        y_m = x.new_zeros((batch_size, self.num_classes))

        for i, layer in enumerate(self.layers):
            latent = layer(latent, x, x, mask)

            # calculate halting probability for current layer
            layer_lambda = (
                x.new_ones((batch_size,))
                if i == len(self.layers) - 1
                else self.lambda_layer(torch.mean(latent, dim=1))
            )
            # calculate current prediction from current layer
            layer_predictions = self.output_layer(torch.mean(latent, dim=1))

            # conditional halting probability for current layer: previously not halted * halting now
            layer_halted_prob = un_halted_prob * layer_lambda.view(-1)
            un_halted_prob = un_halted_prob * (1 - layer_lambda.view(-1))

            # Halt based on the halting probability
            sampling = torch.bernoulli(layer_lambda.reshape(-1))
            halt = sampling * (1 - halted)

            probas.append(layer_halted_prob)
            preds.append(layer_predictions)

            p_m = p_m * (1 - halt) + layer_halted_prob * halt

            y_m = y_m * repeat(
                1 - halt, "B -> B C", C=self.num_classes
            ) + layer_predictions * repeat(halt, "B -> B C", C=self.num_classes)

            halted = halted + halt

            if not self.training and halted.sum() == batch_size:
                break

        return torch.stack(probas), torch.stack(preds), p_m, y_m


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_fn: Callable):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, probas, preds, labels):

        total = preds.new_tensor(0.0)
        for layer_probas, layer_preds in zip(probas, preds):
            layer_loss = layer_probas * self.loss_fn(layer_preds, labels)
            total = total + layer_loss.mean()

        return total


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p: float, max_layers: int):
        super().__init__()
        p_g = torch.zeros((max_layers,))
        not_halted = 1.0
        for k in range(max_layers):
            p_g[k] = lambda_p * not_halted
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, probas):
        probas = probas.transpose(0, 1)
        p_g = self.p_g[None, : probas.shape[1]].expand_as(probas)

        return self.kl_div(probas.log(), p_g)


class XLoss(nn.Module):
    def __init__(self, loss_fn: Callable, lambda_p: float, max_layers: int):
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss(loss_fn)
        self.regularization_loss = RegularizationLoss(lambda_p, max_layers)

    def forward(self, probas, preds, labels):

        return self.reconstruction_loss(
            probas, preds, labels
        ) + self.regularization_loss(probas)
