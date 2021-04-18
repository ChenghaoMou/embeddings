#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-17 08:29:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import pytest
import tempfile
from text_embeddings.hash import CANINETokenizer, PQRNNTokenizer
from transformers.tokenization_utils_base import *

def test_canine_tokenizer():

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
        text_pair=data,
        add_special_tokens=True,
        stride=5,
        padding=PaddingStrategy.LONGEST, 
        return_tensors='pt',
        truncation=TruncationStrategy.LONGEST_FIRST, 
        return_attention_mask=True, 
        return_special_tokens_mask=True,
        return_length=True,
        prepend_batch_axis=True,
        return_overflowing_tokens=False,
    )

    sequence_length = results["input_ids"].shape[1]

    assert sequence_length <= embedder.max_length
    assert results["special_tokens_mask"].shape == (3, sequence_length)
    assert results["input_ids"].shape == (3, sequence_length, 768)
    assert results["length"].shape == (3, )
    assert results["token_type_ids"].shape == (3, sequence_length)

def test_pqrnn_tokenizer():

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
        text_pair=data,
        add_special_tokens=True,
        stride=5,
        padding=PaddingStrategy.LONGEST, 
        return_tensors='pt',
        truncation=TruncationStrategy.LONGEST_FIRST, 
        return_attention_mask=True, 
        return_special_tokens_mask=True,
        return_length=True,
        prepend_batch_axis=True,
        return_overflowing_tokens=False,
    )

    sequence_length = results["input_ids"].shape[1]

    assert sequence_length <= embedder.max_length
    assert results["special_tokens_mask"].shape == (3, sequence_length)
    assert results["input_ids"].shape == (3, sequence_length, 768)
    assert results["length"].shape == (3, )
    assert results["token_type_ids"].shape == (3, sequence_length)