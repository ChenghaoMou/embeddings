import math
from collections import Counter

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from einops import rearrange
from loguru import logger
from sklearn.model_selection import train_test_split as tts
from spacy.lang.en import English
from text_embeddings.visual import VTRTokenizer
from torch.optim import Adam
from torch.utils.data import (BatchSampler, DataLoader, Dataset,
                              SequentialSampler)
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def gen_no_peek_mask(length: int) -> np.ndarray:
    """Generate an N by N mask for autoregressive attention.

    Parameters
    ----------
    length : int
        Length of the sequence

    Returns
    -------
    nd.ndarray
        An N by N mask where allowed positions are marked 
        as zeros while others are negative infinities
    """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, "h w -> w h")
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class Translator(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 10000 + 4,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 96,
        pos_dropout: float = 0.1,
        trans_dropout: float = 0.1,
        warmup_steps: int = 4000,
        lr: float = 1e-09,
    ):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            trans_dropout,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        self.init_weights()

    def init_weights(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        tgt_mask,
    ):
        src = rearrange(src, "n s h w -> s n h w")
        tgt = rearrange(tgt, "n t -> t n")

        src = self.pos_enc(rearrange(src, "s n h w -> s n (h w)"))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        output = rearrange(output, "t n e -> n t e")

        return self.fc(output)

    def training_step(self, batch, batch_idx):

        src, src_key_padding_mask, tgt, tgt_key_padding_mask, ratio = batch
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_mask = gen_no_peek_mask(tgt_inp.shape[1]).to(self.device)

        outputs = self.forward(
            src,
            tgt_inp,
            src_key_padding_mask,
            tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask,
            tgt_mask,
        )
        loss = self.loss(
            rearrange(outputs, "b t v -> (b t) v"), rearrange(tgt_out, "b o -> (b o)")
        )
        self.log("batch_ratio", ratio, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        src, src_key_padding_mask, tgt, tgt_key_padding_mask, _ = batch
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_mask = gen_no_peek_mask(tgt_inp.shape[1]).to(self.device)

        outputs = self.forward(
            src,
            tgt_inp,
            src_key_padding_mask,
            tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask,
            tgt_mask,
        )
        loss = self.loss(
            rearrange(outputs, "b t v -> (b t) v"), rearrange(tgt_out, "b o -> (b o)")
        )

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> None:

        loss = torch.sum(torch.stack([o["val_loss"] for o in outputs]))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        logger.debug(
            self.decode("Strategie republikánské strany proti Obamovu znovuzvolení")
        )

    def configure_optimizers(self):
        print(self.lr)
        optimizer = Adam(self.parameters(), betas=(0.9, 0.98), lr=self.lr, eps=1e-9)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, 18020)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "lr",
            }
        ]

    def decode(self, sentence):
        source_tokenizer = self.trainer.datamodule.tokenizer
        src = source_tokenizer.text2embeddings(sentence)
        src = rearrange(
            torch.from_numpy(src).to(self.device).unsqueeze(0), "n s h w -> s n h w"
        )
        src = self.pos_enc(rearrange(src, "s n h w -> s n (h w)"))

        memory = self.transformer.encoder(src)

        ids = [2]
        tgt = torch.from_numpy(np.asarray([ids])).long().to(self.device)

        while True:
            tgt = rearrange(tgt, "n t -> t n")
            tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
            output = self.transformer.decoder(
                tgt, memory, tgt_mask=gen_no_peek_mask(tgt.shape[0]).to(self.device)
            )
            output = rearrange(output, "t n e -> n t e")
            logits = self.fc(output)
            idx = torch.argmax(logits[0], dim=-1)[-1].item()
            if idx == 3 or len(ids) == self.max_seq_length:
                break
            ids.append(idx)
            tgt = torch.from_numpy(np.asarray([ids])).long().to(self.device)

        return " ".join(
            [self.trainer.datamodule.idx2token.get(id, "[unk]") for id in ids]
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        font_path="/home/chenghaomou/embeddings/Noto_Sans/NotoSans-Regular.ttf",
        font_size: int = 16,
        window_size: int = 8,
        stride: int = 5,
        test_size: float = 0.2,
        max_seq_length: int = 96,
        vocab_size: int = 10000,
        batch_size: int = 2048,
    ):
        super().__init__()
        dataset = load_dataset("wmt14", "cs-en")
        sentences = [
            (x["translation"]["cs"], x["translation"]["en"])
            for x in dataset["validation"]
        ] + [(x["translation"]["cs"], x["translation"]["en"]) for x in dataset["test"]]
        source_tokenizer = VTRTokenizer(
            font=font_path, window_size=window_size, font_size=font_size, stride=stride
        )
        source, target = zip(*sentences)

        nlp = English()
        target_tokenizer = nlp.tokenizer
        target_tokens = [
            [t.text for t in d] for d in tqdm(target_tokenizer.pipe(target))
        ]
        target_vocab = {
            t: i + 4
            for i, (t, _) in enumerate(
                Counter([t for doc in target_tokens for t in doc]).most_common(
                    vocab_size
                )
            )
        }
        target_vocab["[pad]"] = 0
        target_vocab["[oov]"] = 1
        target_vocab["[bos]"] = 2
        target_vocab["[eos]"] = 3

        ids = [
            (
                np.asarray(source_tokenizer.text2embeddings(s)),
                [2] + [target_vocab.get(x, 1) for x in t] + [3],
            )
            for s, t in tqdm(zip(source, target_tokens))
        ]
        ids = [
            (x, y)
            for x, y in ids
            if len(x) <= max_seq_length and len(y) <= max_seq_length
        ]

        self.ids = ids
        self.test_size = test_size
        self.batch_size = batch_size
        self.font_size = font_size
        self.window_size = window_size
        self.tokenizer = source_tokenizer
        self.idx2token = {i: t for t, i in target_vocab.items()}

    def setup(self, stage=None):
        self.train, self.val = tts(self.ids, test_size=self.test_size)

    def train_dataloader(self):
        return DataLoader(
            DummyDataset(self.train),
            batch_sampler=DummySampler(
                SequentialSampler(
                    sorted(range(len(self.train)), key=lambda x: len(self.train[x][0]))
                ),
                batch_size=self.batch_size,
                drop_last=False,
                ids=self.train,
            ),
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            DummyDataset(self.val),
            batch_sampler=DummySampler(
                SequentialSampler(
                    sorted(range(len(self.val)), key=lambda x: len(self.val[x][0]))
                ),
                batch_size=self.batch_size,
                drop_last=False,
                ids=self.val,
            ),
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def collate_fn(self, batch):
        source_input_ids = np.zeros(
            (
                len(batch),
                max(map(lambda x: len(x[0]), batch)),
                self.font_size,
                self.window_size,
            )
        )
        target_input_ids = np.zeros((len(batch), max(map(lambda x: len(x[1]), batch))))
        source_mask = np.zeros(
            (len(batch), max(map(lambda x: len(x[0]), batch))), dtype=bool
        )
        target_mask = np.zeros_like(target_input_ids, dtype=bool)

        for i, (source, target) in enumerate(batch):
            source_input_ids[i, : len(source), :, :] = source
            target_input_ids[i, : len(target)] = target
            source_mask[i, len(source) :] = True
            target_mask[i, len(target) :] = True

        return (
            torch.from_numpy(source_input_ids).float(),
            torch.from_numpy(source_mask),
            torch.from_numpy(target_input_ids).long(),
            torch.from_numpy(target_mask),
            np.count_nonzero((~source_mask).astype(int)) / source_mask.size,
        )


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class DummySampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, ids):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        batch = []
        batches = []
        curr_max = 0
        for idx in self.sampler:
            curr_token = len(ids[idx][0]) + len(ids[idx][1])
            curr_max = max(curr_max, curr_token)
            if curr_max * len(batch) >= self.batch_size:
                batches.append(batch[:])
                batch = [idx]
                curr_max = curr_token
            else:
                batch.append(idx)
        if batch and not self.drop_last:
            batches.append(batch[:])

        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


if __name__ == "__main__":

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(Translator, TranslationDataModule)