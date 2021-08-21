#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-17 08:29:54
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from pathlib import Path

import pytest
import tempfile
from loguru import logger
from text_embeddings.visual import text2image, VTRTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

font_path = str(
    Path(__file__).parent.parent / "resources/Noto_Sans/NotoSans-Regular.ttf"
)
logger.debug(f"Using font_path: {font_path}")


@pytest.mark.parametrize(("text",), [("Hello world!",), ("Hóla!",)])
def test_text2image(text: str):

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img = text2image(text, font=font_path)
        img.save(tmp.name)


@pytest.mark.parametrize(("text",), [("Hello world!",), ("Hóla!",)])
def test_text2embeddings(text: str):

    embedder = VTRTokenizer(font_size=14, window_size=10, font=font_path, max_length=36)

    print(embedder.text2embeddings(text))


@pytest.mark.parametrize(
    (
        "text_pair",
        "add_special_tokens",
        "stride",
        "padding",
        "truncation",
        "return_attention_mask",
        "return_special_tokens_mask",
        "return_length",
    ),
    [
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, True, True),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, True, False),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
        (True, True, 5, PaddingStrategy.LONGEST, "longest_first", False, True, True),
        (True, False, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
        (False, False, 5, PaddingStrategy.LONGEST, "longest_first", True, False, True),
        # (True, True, 5, PaddingStrategy.DO_NOT_PAD, "longest_first", True, True, True),
    ],
)
def test_vtr_tokenizer(
    text_pair: bool,
    add_special_tokens: bool,
    stride: int,
    padding,
    truncation,
    return_attention_mask,
    return_special_tokens_mask,
    return_length,
):

    data = [
        "Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world! Hello world!",
        "Hóla!",
        "你好，世界！",
    ]

    embedder = VTRTokenizer(font_size=14, window_size=10, font=font_path, max_length=36)

    results = embedder(
        text=data,
        text_pair=data if text_pair else None,
        add_special_tokens=add_special_tokens,
        stride=stride,
        padding=padding,
        return_tensors="pt",
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

    if add_special_tokens:
        assert results["input_ids"].shape == (
            3,
            sequence_length,
            14,
            10,
        )  # hight is slightly different because of the font
    else:
        assert results["input_ids"].shape == (
            3,
            sequence_length,
            14,
            10,
        )  # hight is slightly different because of the font
    if return_length:
        assert results["length"].shape == (3,)
    assert results["token_type_ids"].shape == (3, sequence_length)
