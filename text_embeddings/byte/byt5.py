#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-02 08:10:13
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""From ByT5: Towards a token-free future with pre-trained byte-to-byte models."""

import numpy as np
from typing import Optional, List, Dict
from text_embeddings.base import EmbeddingTokenizer
from loguru import logger


class ByT5Tokenizer(EmbeddingTokenizer):
    """Embed text into byte sequences. This is different from other tokenizers because it still has a small vocabulary where each byte is mapped to an index.

    Parameters
    ----------
    embed_size : int, optional
        The size of the embedding, by default 259 (256 + 3 special tokens)
    model_input_names : Optional[List[str]], optional
        Required inputs of the downstream model, by default it uses the same names as a BERT — ["input_ids", "token_type_ids", "attention_mask"]
    special_tokens : Optional[Dict[str, np.ndarray]], optional
        Special tokens for the downstream model, by default it uses the same special tokens as a BERT — {"CLS": "[CLS]", "SEP": "[SEP]"}
    max_length : Optional[int], optional
        Maximum character length, by default 1024

    Examples
    --------
    >>> tokenizer = ByT5Tokenizer()
    >>> e = tokenizer.text2embeddings("This is a test message")
    >>> e.shape
    (22, 259)
    >>> np.equal(np.max(e, axis=1), np.ones((len(e)))).all()
    True
    """

    def __init__(
        self,
        embed_size: int = 259,
        model_input_names: Optional[List[str]] = None,
        special_tokens: Optional[Dict[str, np.ndarray]] = None,
        max_length: Optional[int] = 1024,
    ):
        super().__init__(model_input_names, special_tokens, max_length)
        self.embed_size = embed_size
        self.model_input_names = model_input_names
        self.special_tokens = special_tokens
        self.max_length = max_length

        if self.model_input_names is None:
            logger.warning(
                'Using default model_input_names values ["input_ids", "token_type_ids", "attention_mask"]'
            )
            self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

        if self.special_tokens is None:
            logger.warning("Using default special_tokens values")
            self.special_tokens = {
                "SEP": np.zeros((self.embed_size,)),
                "CLS": np.zeros((self.embed_size,)),
            }
            self.special_tokens["CLS"][1] = 1
            self.special_tokens["SEP"][2] = 1

        logger.info("Be sure to add an embedding layer when using a ByT5Tokenizer.")

    def text2embeddings(self, text: str) -> np.ndarray:
        """Convert text into an numpy array, in (sequence_length, embed_size) shape.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        np.ndarray
            An array in (sequence_length, embed_size) shape
        """
        if not text:
            return None

        b = text.encode("utf-8", errors="ignore")

        result = np.zeros((len(b), self.embed_size))
        for i, byte in enumerate(b):
            result[i][byte + 3] = 1

        return result

    def create_padding_token_embedding(self, input_embeddings=None) -> np.ndarray:
        """Create a padding token embedding.

        Parameters
        ----------
        input_embeddings : np.ndarray, optional
            Embedded input, by default None

        Returns
        -------
        np.ndarray
            A padding token embedding compatible with the input
        """
        e = np.zeros((self.embed_size,))
        e[0] = 1
        return e
