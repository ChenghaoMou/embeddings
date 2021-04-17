![banner](./banner.png)
[![PyPI version](https://badge.fury.io/py/text-embeddings.svg)](https://badge.fury.io/py/text-embeddings) ![Coverage](./coverage.svg)

## Embeddings

- [Visual Text Representations](https://t.co/l9E6rL8O5p?amp=1)

## TODO

- Word-level LSH Embeddings (PRADO/PQRNN)
- Char-level Hash Embeddings (CANINE)

## Installation
```bash
pip install text-embeddings --upgrade
```

## Usage

```python
from text_embeddings.visual import VTRTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

data = [
  "Hello world!",
  "¡Hola Mundo!",
  "你好，世界！",
]

tokenizer = VTRTokenizer(
    height=14,
    width=10,
    font="~/Library/Fonts/NotoSansDisplay-Regular.ttf", # Any font that covers your dataset
    max_length=36
)
results = tokenizer(
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

# (batch_size, sequence_length, height, width) actual height might be higher than the font height value because of rendering
assert results["input_ids"].shape == (3, results["input_ids"].shape[1], 19, 10) 
assert results["attention_mask"].shape == (3, results["input_ids"].shape[1])
assert results["token_type_ids"].shape == (3, results["input_ids"].shape[1])
assert results["length"].shape == (3, )
```
