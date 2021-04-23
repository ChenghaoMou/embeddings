from text_embeddings.visual import VTRTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

def test_readme():

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