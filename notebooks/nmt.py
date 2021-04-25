import torch
import torch.nn as nn
import math

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers.tokenization_utils_base import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MinimalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        target_vocab_length: int = 60000,
    ) -> None:
        super(MinimalTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.target_embedding = nn.Embedding(target_vocab_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )
        self.out = nn.Linear(d_model, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class MinimalTranslator(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        target_vocab_length: int = 60000,
        lr: float = 0.1,
        conv_kernel_size=3,
        use_src_embedding: bool = False,
        source_vocab_length: int = 60000,
        src_tokenizer=None,
        tgt_tokenizer=None,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(source_vocab_length, d_model)
        self.conv_block = nn.ModuleList(
            [
                nn.Conv2d(1, 1, conv_kernel_size, padding_mode="replicate"),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            ]
        )

        self.model = MinimalTransformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            target_vocab_length,
        )
        self.lr = lr
        self.use_src_embedding = use_src_embedding
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        target_input_ids,
        target_token_type_ids,
        target_attention_mask,
    ):

        if not self.use_src_embedding:
            src, tgt = input_ids.float(), target_input_ids
            batch_size, sequence_length = src.shape[0], src.shape[1]

            src = src.view(-1, src.shape[2], src.shape[3])
            src = src.unsqueeze(
                dim=1
            )  # add a channel dimension -> [-1, 1(channel), height, width]
            for i, layer in enumerate(self.conv_block):
                src = layer(src)

            src = src.view((batch_size, sequence_length, -1))
        else:
            src, tgt = input_ids, target_input_ids
            src = self.src_embedding(src)

        src = self.model.pos_encoder(src)

        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:].contiguous().view(-1)
        src_mask = attention_mask != 0
        src_mask = (
            src_mask.float()
            .masked_fill(src_mask == 0, float("-inf"))
            .masked_fill(src_mask == 1, float(0.0))
        )
        tgt_mask = target_attention_mask != 0
        tgt_mask = (
            tgt_mask.float()
            .masked_fill(tgt_mask == 0, float("-inf"))
            .masked_fill(tgt_mask == 1, float(0.0))
        )
        size = tgt_input.size(1)
        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        np_mask = (
            np_mask.float()
            .masked_fill(np_mask == 0, float("-inf"))
            .masked_fill(np_mask == 1, float(0.0))
        )
        preds = self.model(
            src.transpose(0, 1), tgt_input.transpose(0, 1), tgt_mask=np_mask
        )
        preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))
        loss = F.cross_entropy(preds, targets, ignore_index=0, reduction="mean")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(**batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(**batch)
        return {
            "val_loss": loss * len(batch["input_ids"]),
            "size": len(batch["input_ids"]),
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        total = sum([o["size"] for o in outputs])
        loss = sum([o["val_loss"] for o in outputs])
        self.log("val_loss", loss / total, prog_bar=True)

        if self.src_tokenizer and self.tgt_tokenizer:
            print(
                self.greedy_decode_sentence(
                    self,
                    self.src_tokenizer,
                    self.tgt_tokenizer,
                    '"परीक्षण टायरहरू" सबै उत्पादक-निर्भर छन्।',
                )
            )

    @staticmethod
    def greedy_decode_sentence(
        model, src_tokenizer, tgt_tokenizer, sentence: str
    ) -> str:

        model.eval()
        inputs = src_tokenizer(
            [sentence],
            return_tensors="pt",
            return_attention_mask=True,
            padding=PaddingStrategy.LONGEST,
            truncation=TruncationStrategy.LONGEST_FIRST,
        )

        if not model.use_src_embedding:
            src = inputs["input_ids"].float()
            batch_size, sequence_length = src.shape[0], src.shape[1]

            src = src.view(-1, src.shape[2], src.shape[3])
            src = src.unsqueeze(
                dim=1
            )  # add a channel dimension -> [-1, 1(channel), height, width]
            for i, layer in enumerate(model.conv_block):
                src = layer(src)

            src = src.view((batch_size, sequence_length, -1))
        else:
            src = inputs["input_ids"]
            src = model.src_embedding(src)

        src = model.model.pos_encoder(src)

        trg = torch.LongTensor([[tgt_tokenizer.cls_token_id]])
        translated_sentence = ""
        maxlen = 50
        for i in range(maxlen):
            size = trg.size(0)
            np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
            np_mask = (
                np_mask.float()
                .masked_fill(np_mask == 0, float("-inf"))
                .masked_fill(np_mask == 1, float(0.0))
            )

            pred = model.model(src.transpose(0, 1), trg, tgt_mask=np_mask)
            add_word = tgt_tokenizer.ids_to_tokens.get(
                pred.argmax(dim=2)[-1], tgt_tokenizer.unk_token
            )
            translated_sentence += " " + add_word
            if add_word == tgt_tokenizer.sep_token:
                break
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]])))

        return translated_sentence

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class MTDataset(Dataset):
    def __init__(self, src, tgt):
        self.src_data = [line for line in Path(src).read_text().split("\n") if line]
        self.tgt_data = [line for line in Path(tgt).read_text().split("\n") if line]

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, index):
        return {"source": self.src_data[index], "target": self.src_data[index]}


def collate_fn(data, src_tokenizer, tgt_tokenizer):

    source = [d["source"] for d in data]
    target = [d["target"] for d in data]

    results = {}
    if not source or not target:
        raise ValueError("Source and target should not be empty")
    for key, result in src_tokenizer(
        source,
        return_tensors="pt",
        return_attention_mask=True,
        padding=PaddingStrategy.LONGEST,
        truncation=TruncationStrategy.LONGEST_FIRST,
    ).items():
        results[key] = result
    for key, result in tgt_tokenizer(
        target,
        return_tensors="pt",
        return_attention_mask=True,
        padding=PaddingStrategy.LONGEST,
        truncation=TruncationStrategy.LONGEST_FIRST,
    ).items():
        results["target_" + key] = result

    return results


if __name__ == "__main__":

    from text_embeddings.visual import VTRTokenizer
    from transformers import AutoTokenizer
    from functools import partial
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    window_size = 10
    font_size = 14
    conv_kernel_size = 3
    use_src_embedding = False

    if not use_src_embedding:
        src_tokenizer = VTRTokenizer(
            window_size=window_size,
            font_size=font_size,
        )
    else:
        src_tokenizer = AutoTokenizer.from_pretrained(
            "sagorsarker/codeswitch-nepeng-lid-lince"
        )
    tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

    collate_fn_partial = partial(
        collate_fn, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )

    train_dataloader = DataLoader(
        MTDataset(
            "/Users/chenghaomou/Downloads/flores_test_sets/wikipedia.dev.ne-en.ne",
            "/Users/chenghaomou/Downloads/flores_test_sets/wikipedia.dev.ne-en.en",
        ),
        batch_size=4,
        collate_fn=collate_fn_partial,
        num_workers=24,
    )
    dev_dataloader = DataLoader(
        MTDataset(
            "/Users/chenghaomou/Downloads/flores_test_sets/wikipedia.test.ne-en.ne",
            "/Users/chenghaomou/Downloads/flores_test_sets/wikipedia.test.ne-en.en",
        ),
        batch_size=4,
        collate_fn=collate_fn_partial,
        num_workers=24,
    )

    trainer = pl.Trainer(
        deterministic=True,
        val_check_interval=0.2,
        gradient_clip_val=1.0,
    )
    model = MinimalTranslator(
        d_model=(window_size - conv_kernel_size + 1)
        * (font_size - conv_kernel_size + 1),
        target_vocab_length=tgt_tokenizer.vocab_size,
        conv_kernel_size=conv_kernel_size,
        use_src_embedding=use_src_embedding,
        source_vocab_length=getattr(src_tokenizer, "vocab_size", 10),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )
    trainer.fit(model, train_dataloader, dev_dataloader)
