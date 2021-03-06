![banner](./banner.png)
[![PyPI version](https://badge.fury.io/py/text-embeddings.svg)](https://badge.fury.io/py/text-embeddings) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/112e50abd97444a4aca06f94fb7e8873)](https://www.codacy.com/gh/ChenghaoMou/embeddings/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/embeddings&utm_campaign=Badge_Grade)[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/112e50abd97444a4aca06f94fb7e8873)](https://www.codacy.com/gh/ChenghaoMou/embeddings/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ChenghaoMou/embeddings&utm_campaign=Badge_Coverage)

## Features

-   [x] `VTRTokenizer` from [Robust Open­-Vocabulary Translation from Visual Text Representations](https://t.co/l9E6rL8O5p?amp=1)
-   [x] `PQRNNTokenizer` from [Advancing NLP with Efficient Projection-Based Model Architectures](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html)
-   [x] `CANINETokenizer` from [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874)
-   [x] `ByT5Tokenizer` from [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/pdf/2105.13626.pdf)
-   [x] `GBST` and `ByteTokenizer` from [Charformer: Fast Character Transformers via Gradient-based Subword Tokenization](https://arxiv.org/abs/2106.12672)
-   [x] `LTPMultiHeadAttention` from [Learned Token Pruning for Transformers](https://arxiv.org/abs/2107.00910)
-   [x] `X` and `XLoss`, a model inspired from [PonderNet](https://arxiv.org/abs/2107.05407) and [Perceiver](https://arxiv.org/abs/2103.03206), with Byte Embeddings.

## Examples

-   [x] [Machine Translation](examples/translation/nmt_transformer.py)
-   [x] [Text Classification](examples/classification/rnn.py)

## Installation

```bash
pip install text-embeddings --upgrade
```

## Documentation

[Link](https://chenghaomou.github.io/embeddings/)

## Example Usage

```python
from text_embeddings.visual import VTRTokenizer

data = [
"Hello world!",
"¡Hola Mundo!",
"你好，世界！",
]

tokenizer = VTRTokenizer(
    font_size=14,
    window_size=10,
    font="resources/NotoSans-Regular.ttf",
    max_length=36
)

results = tokenizer(
    text=data,
    text_pair=data,
    add_special_tokens=True,
    padding="longest", 
    return_tensors='pt',
    truncation="longest_first", 
    return_attention_mask=True, 
    return_special_tokens_mask=True,
    return_length=True,
    prepend_batch_axis=True,
    return_overflowing_tokens=False,
)

assert results["input_ids"].shape == (3, results["input_ids"].shape[1], 14, 10) 
assert results["attention_mask"].shape == (3, results["input_ids"].shape[1])
assert results["token_type_ids"].shape == (3, results["input_ids"].shape[1])
assert results["length"].shape == (3, )
```

## Write Your Own Embedding Tokenizer

```python
import numpy as np
from typing import Optional, List, Dict
from text_embeddings.base import EmbeddingTokenizer


class MyOwnTokenizer(EmbeddingTokenizer):

    def __init__(
        self,
        model_input_names: Optional[List[str]] = None,
        special_tokens: Optional[Dict[str, np.ndarray]] = None,
        max_length: Optional[int] = 2048,
    ):
        super().__init__(model_input_names, special_tokens, max_length)

    def text2embeddings(self, text: str) -> np.ndarray:
        
        sequence_length = 10
        dimensions = (10, 10, 10) # each token is mapped to a 3-d array
        return np.zeros((sequence_length, *dimensions))

    def create_padding_token_embedding(self, input_embeddings=None) -> np.ndarray:

        # let's create a consistent 3-d array
        return np.zeros((10, 10, 10))

```

## Example Usage for GBST

```python
import torch.onnx  # nightly torch only
from text_embeddings.byte.charformer import GBST, ByteTokenizer

model = GBST(
    embed_size=128,
    max_block_size=4,
    downsampling_factor=2,
    score_calibration=True,
    vocab_size=259,
)

tokenizer = ByteTokenizer()
results = tokenizer(
    ["Life is like a box of chocolates.", "Coding is fun."],
    add_special_tokens=True,
    padding="longest",
    truncation="longest_first",
)

# Export the model
torch.onnx.export(
    model,
    torch.tensor(results["input_ids"], requires_grad=True).long(),
    "gbst.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size"},
    },
)
```
