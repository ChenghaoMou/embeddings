#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-18 09:06:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

"""https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html"""

import numpy as np
from typing import Optional, List, Dict
from text_embeddings.hash.util import murmurhash
from text_embeddings.base import EmbeddingTokenizer
from loguru import logger

class PQRNNTokenizer(EmbeddingTokenizer):
    """
    Boundary-based hashing embeddings based on [PQRNN](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html)

    Parameters
    ----------
    hash_size : int, optional
        The size of the hashing embedding, by default 768
    model_input_names : Optional[List[str]], optional
        Required inputs of the downstream model, by default it uses the same names as a BERT — ["input_ids", "token_type_ids", "attention_mask"]
    special_tokens : Optional[Dict[str, np.ndarray]], optional
        Special tokens for the downstream model, by default it uses the same special tokens as a BERT — {"CLS": "[CLS]", "SEP": "[SEP]"}
    max_length : Optional[int], optional
        Maximum token length, by default 2048
    
    Examples
    --------
    >>> from text_embeddings.hash import PQRNNTokenizer
    >>> from transformers.tokenization_utils_base import *
    >>> tokenier = PQRNNTokenizer()
    >>> results = tokenier(text=['This is a sentence.', 'This is another sentence.'], padding=PaddingStrategy.LONGEST, truncation=TruncationStrategy.LONGEST_FIRST, add_special_tokens=False)
    >>> assert results['input_ids'].shape == (2, 4, 768)
    """

    def __init__(
        self,
        hash_size: int = 768,
        model_input_names: Optional[List[str]] = None,
        special_tokens: Optional[Dict[str, np.ndarray]] = None,
        max_length: Optional[int] = 2048,
    ):
        super().__init__(model_input_names, special_tokens, max_length)
        self.hash_size = hash_size
        self.model_input_names = model_input_names
        self.special_tokens = special_tokens
        self.max_length = max_length
        
        if self.model_input_names is None:
            logger.warning('Using default model_input_names values ["input_ids", "token_type_ids", "attention_mask"]')
            self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def text2embeddings(self, text: str) -> np.ndarray:
        """Convert text into an numpy array, in (sequence_length, 1, hash_size) shape.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        np.ndarray
            An array in (sequence_length, 1, hash_size) shape
        """
        if not text:
            return None
        
        tokens = text.split(" ")
        result = np.zeros((len(tokens), self.hash_size))
        for i, token in enumerate(tokens):
            result[i] = murmurhash(token, feature_size=self.hash_size*2)
        
        return result

    def create_padding_token_embedding(self, input_embeddings=None) -> np.ndarray:
        """Create a padding token embedding

        Parameters
        ----------
        input_embeddings : [type], optional
            [description], by default None

        Returns
        -------
        np.ndarray
            An empty embedding in (hash_size, ) shape
        """
        
        return np.zeros((self.hash_size, ))
