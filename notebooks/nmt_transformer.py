import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from einops import rearrange
from torch.utils.data import Dataset, DataLoader

MAX_LENGTH = 256


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Seq2SeqModel(pl.LightningModule):
    def __init__(
        self,
        src_tokenizer,
        trg_tokenizer,
        d_model=512,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=1024,
        max_seq_length=512,
        pos_dropout=0.5,
        trans_dropout=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(src_tokenizer.vocab_size, d_model)
        self.embed_trg = nn.Embedding(trg_tokenizer.vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            trans_dropout,
        )
        self.fc = nn.Linear(d_model, trg_tokenizer.vocab_size)
        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=trg_tokenizer.pad_token_id
        )

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def forward(
        self,
        src,
        trg,
        src_mask,
        trg_mask,
        src_key_padding_mask=None,
        trg_key_padding_mask=None,
    ):
        src = rearrange(src, "b s -> s b")
        trg = rearrange(trg, "b s -> s b")
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        trg = self.pos_enc(self.embed_trg(trg) * math.sqrt(self.d_model))

        output = self.transformer(
            src,
            trg,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
        )
        output = rearrange(output, "s b h -> b s h")
        return self.fc(output)

    def training_step(self, batch, batch_idx):

        src = batch["input_ids"]
        trg = batch["target_input_ids"][:, :-1]

        truth = batch["target_input_ids"][:, 1:]

        teacher_forcing = True if random.uniform(0, 1) <= 1.0 else False

        if teacher_forcing:
            output = self.forward(
                src=src,
                trg=trg,
                src_mask=self.transformer.generate_square_subsequent_mask(
                    src.shape[1]
                ).to(self.device),
                trg_mask=self.transformer.generate_square_subsequent_mask(
                    trg.shape[1]
                ).to(self.device),
                src_key_padding_mask=~batch["attention_mask"].bool(),
                trg_key_padding_mask=~batch["target_attention_mask"][:, :-1].bool(),
            )

            return {
                "loss": self.loss(
                    output.reshape((-1, output.shape[-1])), truth.reshape(-1)
                )
            }
        else:
            src = rearrange(batch["input_ids"], "b s -> s b")
            src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))

            memory = self.transformer.encoder(
                src,
                mask=self.transformer.generate_square_subsequent_mask(
                    batch["input_ids"].shape[1]
                ).to(self.device),
                src_key_padding_mask=~batch["attention_mask"].bool(),
            )

            trg_ids = (
                torch.from_numpy(
                    np.asarray(
                        [
                            [self.trg_tokenizer.cls_token_id]
                            for _ in range(batch["target_input_ids"].shape[0])
                        ]
                    )
                )
                .long()
                .to(self.device)
            )
            loss = 0
            for i in range(batch["target_input_ids"].shape[1] - 1):
                trg = rearrange(trg_ids, "b s -> s b")
                trg = self.pos_enc(self.embed_trg(trg) * math.sqrt(self.d_model))
                output = self.transformer.decoder(
                    trg,
                    memory,
                    tgt_mask=self.transformer.generate_square_subsequent_mask(
                        trg.shape[0]
                    ).to(self.device),
                )
                
                output = self.fc(output)
                loss += self.loss(
                    output[-1].reshape((-1, output.shape[-1])),
                    batch["target_input_ids"][:, i + 1].reshape(-1),
                )
                trg_ids = torch.cat(
                    [trg_ids, torch.argmax(output[-1], dim=-1, keepdim=True)], dim=-1
                )

            return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        src = batch["input_ids"]
        trg = batch["target_input_ids"][:, :-1]

        truth = batch["target_input_ids"][:, 1:]

        output = self.forward(
            src=src,
            trg=trg,
            src_mask=self.transformer.generate_square_subsequent_mask(src.shape[1]).to(
                self.device
            ),
            trg_mask=self.transformer.generate_square_subsequent_mask(trg.shape[1]).to(
                self.device
            ),
            src_key_padding_mask=~batch["attention_mask"].bool(),
            trg_key_padding_mask=~batch["target_attention_mask"][:, :-1].bool(),
        )
        loss = self.loss(output.reshape((-1, output.shape[-1])), truth.reshape(-1))
        self.log("val_loss", loss.item(), prog_bar=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> None:

        inp = "C'est un chat bleu"
        batch = self.src_tokenizer(
            [inp],
            return_tensors="pt",
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
        )

        src = rearrange(batch["input_ids"].to(self.device), "b s -> s b")
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))

        memory = self.transformer.encoder(
            src,
            mask=self.transformer.generate_square_subsequent_mask(
                batch["input_ids"].shape[1]
            ).to(self.device),
            src_key_padding_mask=~batch["attention_mask"].bool().to(self.device),
        )
        ids = [[self.trg_tokenizer.cls_token_id]]
        translation = []
        for i in range(50):
            trg = torch.from_numpy(np.asarray(ids)).long().to(self.device)
            trg = rearrange(trg, "b s -> s b")
            trg = self.pos_enc(self.embed_trg(trg) * math.sqrt(self.d_model))
            output = self.transformer.decoder(
                trg,
                memory,
                tgt_mask=self.transformer.generate_square_subsequent_mask(
                    trg.shape[0]
                ).to(self.device),
            )
            output = self.fc(output)
            pred = torch.argmax(output[-1].view(-1)).item()
            ids[0] = ids[0] + [pred]
            translation.append(pred)
            if pred == self.trg_tokenizer.eos_token_id:
                break
        print(self.trg_tokenizer.decode(translation, skip_special_tokens=True))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return {"optimizer": optimizer, "scheduler": scheduler}


class TranslationDataset(pl.LightningDataModule):
    def __init__(self, data, src_tokenizer, trg_tokenizer, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.train_data = [(r["cs"], r["en"]) for r in data["train"]["translation"]]
        self.val_data = [(r["cs"], r["en"]) for r in data["validation"]["translation"]]

    def setup(self, stage=None):
        self.train = DummyDataset(self.train_data)
        self.val = DummyDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def collate_fn(self, data):

        source = [d["source"] for d in data]
        target = [d["target"] for d in data]

        result = {}
        if source:
            r = self.src_tokenizer(
                source,
                return_tensors="pt",
                padding="longest",
                truncation="longest_first",
                return_attention_mask=True,
                max_length=MAX_LENGTH,
            )
            result["input_ids"] = r.input_ids
            result["attention_mask"] = r.attention_mask

        if target:
            r = self.trg_tokenizer(
                target,
                return_tensors="pt",
                padding="longest",
                truncation="longest_first",
                return_attention_mask=True,
                max_length=MAX_LENGTH,
            )
            result["target_input_ids"] = r.input_ids
            result["target_attention_mask"] = r.attention_mask

        return result


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "source": " ".join(self.data[index][1].split(" ")[::-1]),
            "target": self.data[index][1],
        }


if __name__ == "__main__":

    from transformers import AutoTokenizer
    from datasets import load_dataset

    dataset = load_dataset("wmt14", "cs-en")

    source_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    target_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    trainer = pl.Trainer(
        gpus=1,
        check_val_every_n_epoch=1,
        precision=16,
        accumulate_grad_batches=4,
        val_check_interval=0.1,
    )
    model = Seq2SeqModel(
        src_tokenizer=source_tokenizer,
        trg_tokenizer=target_tokenizer,
        max_seq_length=MAX_LENGTH,
    )
    data = TranslationDataset(
        data=dataset,
        src_tokenizer=source_tokenizer,
        trg_tokenizer=target_tokenizer,
        batch_size=64,
    )
    trainer.fit(model, data)