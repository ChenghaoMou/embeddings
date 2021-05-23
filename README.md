![banner](./banner.png)
[![PyPI version](https://badge.fury.io/py/text-embeddings.svg)](https://badge.fury.io/py/text-embeddings) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/112e50abd97444a4aca06f94fb7e8873)](https://www.codacy.com/gh/ChenghaoMou/embeddings/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ChenghaoMou/embeddings&amp;utm_campaign=Badge_Grade)

## Features

-   [x] [Visual Text Representations](https://t.co/l9E6rL8O5p?amp=1)
-   [x] Word-level Hash Embeddings ([PRADO/PQRNN](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html))
-   [x] Char-level Hash Embeddings ([CANINE](https://arxiv.org/abs/2103.06874))

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
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

data = [
"Hello world!",
"¡Hola Mundo!",
"你好，世界！",
]

tokenizer = VTRTokenizer(
    font_size=14,
    window_size=10,
    font="~/Library/Fonts/NotoSansDisplay-Regular.ttf",
    max_length=36
)

results = tokenizer(
    text=data,
    text_pair=data,
    add_special_tokens=True,
    padding=PaddingStrategy.LONGEST, 
    return_tensors='pt',
    truncation=TruncationStrategy.LONGEST_FIRST, 
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
