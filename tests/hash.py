#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-17 08:29:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import pytest
from text_embeddings.hash import CANINETokenizer, PQRNNTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

@pytest.mark.parametrize(
    ('text_pair', 'add_special_tokens', 'stride', 'padding', 'truncation', 'return_attention_mask', 'return_special_tokens_mask', 'return_length'), [
        (True, True, 5, "longest", "longest_first", True, True, True),
        (True, True, 5, "longest", "longest_first", True, True, False),
        (True, True, 5, "longest", "longest_first", True, False, True),
        (True, True, 5, "longest", "longest_first", False, True, True),
        (True, False, 5, "longest", "longest_first", True, False, True),
        (False, False, 5, "longest", "longest_first", True, False, True),
    ]
)
def test_canine_tokenizer(text_pair: bool, add_special_tokens: bool, stride: int, padding, truncation, return_attention_mask, return_special_tokens_mask, return_length):

    data = [
        "Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world!",
        "Hóla!",
        "你好，世界！",
    ]

    embedder = CANINETokenizer(
        hash_size=768,
        max_length=2048
    )

    results = embedder(
        text=data,
        text_pair=data if text_pair else None,
        add_special_tokens=add_special_tokens,
        stride=stride,
        padding=padding,
        return_tensors='pt',
        truncation=truncation,
        return_attention_mask=return_attention_mask,
        return_special_tokens_mask=return_special_tokens_mask,
        return_length=return_length,
        prepend_batch_axis=True,
        return_overflowing_tokens=False,
    )

    sequence_length = results["input_ids"].shape[1]

    assert sequence_length <= embedder.max_length
    if return_special_tokens_mask and add_special_tokens:
        assert results["special_tokens_mask"].shape == (3, sequence_length)

    assert results["input_ids"].shape == (3, sequence_length, 768) # hight is slightly different because of the font
    if return_length:
        assert results["length"].shape == (3, )

@pytest.mark.parametrize(
    ('text_pair', 'add_special_tokens', 'stride', 'padding', 'truncation', 'return_attention_mask', 'return_special_tokens_mask', 'return_length'), [
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, True, True),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, True, False),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", False, True, True),
        (True, False, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
        (False, False, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
    ]
)
def test_pqrnn_tokenizer(text_pair: bool, add_special_tokens: bool, stride: int, padding, truncation, return_attention_mask, return_special_tokens_mask, return_length):

    data = [
        "Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world!",
        "Hóla!",
        "你好，世界！",
    ]

    embedder = PQRNNTokenizer(
        hash_size=768,
        max_length=512
    )

    results = embedder(
        text=data,
        text_pair=data if text_pair else None,
        add_special_tokens=add_special_tokens,
        stride=stride,
        padding=padding,
        return_tensors='pt',
        truncation=truncation,
        return_attention_mask=return_attention_mask,
        return_special_tokens_mask=return_special_tokens_mask,
        return_length=return_length,
        prepend_batch_axis=True,
        return_overflowing_tokens=False,
    )

    sequence_length = results["input_ids"].shape[1]

    assert sequence_length <= embedder.max_length
    if return_special_tokens_mask and add_special_tokens:
        assert results["special_tokens_mask"].shape == (3, sequence_length)

    assert results["input_ids"].shape == (3, sequence_length, 768) # hight is slightly different because of the font
    if return_length:
        assert results["length"].shape == (3, )
    assert results["token_type_ids"].shape == (3, sequence_length)