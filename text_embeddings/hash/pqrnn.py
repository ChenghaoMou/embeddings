#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-04-18 09:06:29
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
from transformers.tokenization_utils_base import *
from itertools import zip_longest
from text_embeddings.hash.util import murmurhash

def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


@dataclass
class PQRNNTokenizer(PreTrainedTokenizerBase):

    hash_size: int = 768
    model_input_names: Optional[List[str]] = None
    special_tokens: Optional[Dict[str, np.ndarray]] = None
    max_length: Optional[int] = 2048

    def __post_init__(self):
        if self.model_input_names is None:
            # Assume the model takes BERT-like parameters
            self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        text_pair: Optional[Union[TextInput, List[TextInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """Tokenize the text into a sequence of image blocks. Reference paper: https://t.co/l9E6rL8O5p?amp=1

        Parameters
        ----------
        text : Union[TextInput, List[TextInput]]
            A single text or a list of text
        text_pair : Optional[Union[TextInput, List[TextInput]]], optional
            A single text or a list of text, by default None
        add_special_tokens : bool, optional
            Whether to add special tokens to the data, by default True
        padding : Union[bool, str, PaddingStrategy], optional
            The padding strategy, by default False
        truncation : Union[bool, str, TruncationStrategy], optional
            The truncation strategy, by default False
        max_length : Optional[int], optional
            Maximum sequence length, overriding the class variable, by default None
        stride : int, optional
            Stride for generating blocks, by default 0
        pad_to_multiple_of : Optional[int], optional
            Padding parameters, by default None
        return_tensors : Optional[Union[str, TensorType]], optional
            Return tensors in `pt`, 'tf' or 'np', by default None
        return_token_type_ids : Optional[bool], optional
            Return token type ids, by default None
        return_attention_mask : Optional[bool], optional
            Return attention mask, by default None
        return_overflowing_tokens : bool, optional
            Return overflowing tokens, by default False
        return_special_tokens_mask : bool, optional
            Return special token mask, by default False
        return_length : bool, optional
            Return length, by default False

        Returns
        -------
        BatchEncoding
            A BatchEncoding object
        """
        if self.special_tokens is None:
            self.special_tokens = {
                "CLS": self.text2hashes("[CLS]"),
                "SEP": self.text2hashes("[SEP]"),
            }

        if add_special_tokens and text_pair:
            actual_max_length = self.max_length - len(self.special_tokens["SEP"]) * 2 - len(self.special_tokens["CLS"])
        else:
            actual_max_length = self.max_length
        
        batch_outputs = {}
        text = text if isinstance(text, list) else [text]
        text_pair = text_pair if isinstance(text_pair, list) else [text_pair]

        for first_text, second_text in zip_longest(text, text_pair):
            
            first_embeddings = self.text2hashes(first_text)
            second_embeddings = self.text2hashes(second_text)

            outputs = self.prepare_for_model(
                first_embeddings,
                second_embeddings,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                truncation=truncation,
                max_length=max_length or actual_max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding,
            max_length=max_length or actual_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        if "input_ids" in batch_outputs:
            batch_outputs["input_ids"] = np.squeeze(batch_outputs["input_ids"], axis=2)

        return batch_outputs

    def text2hashes(self, text: str) -> np.ndarray:
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
        
        return np.expand_dims(result, axis=1)

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        prepend_batch_axis: bool = False,
        **kwargs
    ):

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
        
        # Build output dictionary
        encoded_inputs["input_ids"] = sequence

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Padding
        if padding != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        
        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )
        
        return batch_outputs
    
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 0 if not pair else 3
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        
        return np.concatenate(
            [
                self.special_tokens["CLS"],
                token_ids_0,
                self.special_tokens["SEP"],
                token_ids_1,
                self.special_tokens["SEP"],
            ],
            axis=0
        )
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [0 for _ in token_ids_0]
        return [1 for _ in self.special_tokens["CLS"]] + [0 for _ in token_ids_0] + [1 for _ in self.special_tokens["SEP"]] + [0 for _ in token_ids_1] + [1 for _ in self.special_tokens["SEP"]]
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0]*len(self.special_tokens["CLS"]) + [0] * len(token_ids_0) + [0]*len(self.special_tokens["SEP"]) + [1] * len(token_ids_1) + [0]*len(self.special_tokens["SEP"])

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy or pytorch object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)
        
        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:

        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if "token_type_ids" in encoded_inputs and isinstance(encoded_inputs["token_type_ids"], int):
                encoded_inputs["token_type_ids"] = [encoded_inputs["token_type_ids"]]
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [1] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                
                encoded_inputs[self.model_input_names[0]] = required_input + [np.zeros((len(required_input[0]), self.hash_size))] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [1] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [np.zeros((len(required_input[0]), self.hash_size))] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        return encoded_inputs