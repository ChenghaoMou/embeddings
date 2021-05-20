#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-04-17 08:08:04
# @Author       : Chenghao Mou (mouchenghao@gmail.com)
# @Description  : From Robust Open-­Vocabulary Translation from Visual Text Representations

"""Robust Open­ Vocabulary Translation from Visual Text Representations"""

from typing import List, Optional, Dict

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy.lib.stride_tricks import sliding_window_view
from loguru import logger

from text_embeddings.base import EmbeddingTokenizer

def text2image(text: str, font: str, font_size: int = 14) -> Image:
    """Convert text into an image and return the image. Reference: https://gist.github.com/destan/5540702

    Parameters
    ----------
    text : str
        Text to encode
    font : str
        Name of the font to use
    font_size : int, optional
        Size of the font, by default 14

    Returns
    -------
    Image
        Encoded image
    """

    image_font = ImageFont.truetype(font, max(font_size - 2, 8))
    text = text.replace("\n", " ")

    line_width, _ = image_font.getsize(text)

    img = Image.new("L", (line_width, font_size))
    draw = ImageDraw.Draw(img)
    draw.text(xy=(0, 0), text=text, fill="#FFFFFF", font=image_font)

    return img

class VTRTokenizer(EmbeddingTokenizer):
    """
    Render the text into a series of image blocks. Reference [VTR](https://t.co/l9E6rL8O5p?amp=1)

    Parameters
    ----------
    window_size : int, optional
        The width of the image window, by default 10
    stride: int optional
        The stride used to generate image windows, by default 10
    font : str, optional
        Path to the font file, by default "~/Library/Fonts/NotoSansDisplay-Regular.ttf"
    font_size : int, optional
        The size of the font in pixels, might be smaller than the actual image height, by default 14
    model_input_names : List[str], optional
        Required inputs of the downstream model, by default it uses the same names as a BERT — ["input_ids", "token_type_ids", "attention_mask"]
    special_tokens : Optional[Dict[str, np.ndarray]], optional
        Special tokens for the downstream model, by default it uses the same special tokens as a BERT — {"CLS": "[CLS]", "SEP": "[SEP]"}
    max_length : Optional[int], optional
        Maximum sequence length, by default 25

    Examples
    --------
    >>> from text_embeddings.visual import VTRTokenizer
    >>> from transformers.tokenization_utils_base import *
    >>> tokenier = VTRTokenizer()
    >>> results = tokenier(text=['This is a sentence.', 'This is another sentence.'], padding=PaddingStrategy.LONGEST, truncation=TruncationStrategy.LONGEST_FIRST, add_special_tokens=False)
    >>> assert results['input_ids'].shape == (2, 13, 14, 10), results['input_ids'].shape
    """

    def __init__(
        self,
        window_size: int = 10,
        stride: int = 10,
        font: str = "~/Library/Fonts/NotoSansDisplay-Regular.ttf",
        font_size: int = 14,
        model_input_names: List[str] = None,
        special_tokens: Optional[Dict[str, np.ndarray]] = None,
        max_length: Optional[int] = 25,
    ):
        super().__init__(model_input_names, special_tokens, max_length)
        self.font_size = font_size
        self.window_size = window_size
        self.stride = stride
        self.font = font

        if self.model_input_names is None:
            logger.warning('Using default model_input_names values ["input_ids", "token_type_ids", "attention_mask"]')
            self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def text2embeddings(self, text: str) -> np.ndarray:
        """Convert text into an numpy array, in (sequence_length, font_size, window_size) shape.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        np.ndarray
            An array in (sequence_length, height, width) shape
        """
        if not text:
            return None

        image = text2image(text, font=self.font, font_size=self.font_size)
        image_array = np.asarray(image)

        return np.squeeze(
            sliding_window_view(image_array, (image_array.shape[0], self.window_size)),
            axis=0,
        )[:: self.stride]

    def create_padding_token_embedding(self, input_embeddings=None) -> np.ndarray:
        """Create a padding token embedding for an empty window.

        Parameters
        ----------
        input_embeddings : [type], optional
            Embeddings already encoded, by default None

        Returns
        -------
        np.ndarray
            An empty array in (font_size, window_size) shape
        """
        return np.zeros((len(input_embeddings[0]), self.window_size))
