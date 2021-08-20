#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-07-18 14:39:02
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LTPMultiHeadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        temperature,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        Examples
        --------
        >>> attention = LTPMultiHeadAttention(1, 512, 8, 0.5, batch_first=True)
        >>> x = torch.rand((10, 128, 512))
        >>> output, weights, norm = attention(x, x, x)
        >>> output.shape
        torch.Size([10, 128, 512])
        >>> weights.shape
        torch.Size([10, 128, 128])
        """
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )
        self.temperature = temperature
        self.soft_threshold = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shapes for inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
              source sequence length.

              If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
              length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
              the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )

        # (N, L, S) -> (N, S/L)
        scores = torch.mean(attn_output_weights, dim=1)
        pruning_mask = F.sigmoid((scores - self.soft_threshold) / self.temperature)
        attn_output = attn_output.transpose(1, 0)
        attn_output = pruning_mask[:, :, None] * attn_output

        if self.batch_first:
            return (
                attn_output,
                attn_output_weights,
                torch.sum(torch.norm(pruning_mask, p=1) / self.num_heads),
            )
        else:
            return (
                attn_output.transpose(1, 0),
                attn_output_weights,
                torch.sum(torch.norm(pruning_mask, p=1) / self.num_heads),
            )
