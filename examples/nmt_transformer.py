import math
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import spacy
from tqdm import tqdm
from einops import rearrange
from torch.optim import Adam
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler
from sklearn.model_selection import train_test_split as tts
from transformers import get_cosine_schedule_with_warmup

def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

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
        self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        self.init_weights()
    
    def init_weights(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        src = rearrange(src, 'n s -> s n')
        tgt = rearrange(tgt, 'n t -> t n')

        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        output = rearrange(output, 't n e -> n t e')

        return self.fc(output)
    
    def training_step(self, batch, batch_idx):
        
        src, src_key_padding_mask, tgt, tgt_key_padding_mask, ratio = batch
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(self.device)

        outputs = self.forward(
            src, 
            tgt_inp, 
            src_key_padding_mask, 
            tgt_key_padding_mask[:, :-1], 
            memory_key_padding_mask, 
            tgt_mask
        )
        loss = self.loss(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
        # print(batch)
        self.log("batch_ratio", ratio, prog_bar=True)
        return {
            "loss": loss
        }
    
    def validation_step(self, batch, batch_idx):
        
        src, src_key_padding_mask, tgt, tgt_key_padding_mask, ratio = batch
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(self.device)

        outputs = self.forward(
            src, 
            tgt_inp, 
            src_key_padding_mask, 
            tgt_key_padding_mask[:, :-1], 
            memory_key_padding_mask, 
            tgt_mask
        )
        loss = self.loss(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

        return {
            "val_loss": loss
        }
    
    def validation_epoch_end(self, outputs) -> None:
        
        loss = torch.sum(torch.stack([o["val_loss"] for o in outputs]))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        print(self.lr)
        optimizer = Adam(self.parameters(), betas=(0.9, 0.98), lr=self.lr, eps=1e-9)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, 18020)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'lr',
        }]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TranslationDataModule(pl.LightningDataModule):

    def __init__(self, data_path, test_size: float = 0.2, max_seq_length: int = 96, vocab_size: int = 10000, batch_size: int = 2048):
        super().__init__()
        sentences = [x.split('\t')[:2] for x in Path(data_path).read_text().splitlines(keepends=True)]
        source, target = zip(*sentences)
        fr_nlp = spacy.load("fr_core_news_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        en_nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        source_tokens = [[t.text for t in d] for d in tqdm(en_nlp.pipe(source))]
        target_tokens = [[t.text for t in d] for d in tqdm(fr_nlp.pipe(target))]

        source_vocab = {t: i + 4 for i, (t, _) in enumerate(Counter([t for doc in source_tokens for t in doc]).most_common(vocab_size))}
        target_vocab = {t: i + 4 for i, (t, _) in enumerate(Counter([t for doc in target_tokens for t in doc]).most_common(vocab_size))}

        source_vocab['[pad]'] = target_vocab['[pad]'] = 0
        source_vocab['[oov]'] = target_vocab['[oov]'] = 1
        source_vocab['[bos]'] = target_vocab['[bos]'] = 2
        source_vocab['[eos]'] = target_vocab['[eos]'] = 3

        ids = [([source_vocab.get(x, 1) for x in s], [2] + [target_vocab.get(x, 1) for x in t] + [3]) for s, t in zip(source_tokens, target_tokens)]
        ids = [(x, y) for x, y in ids if len(x) <= max_seq_length and len(y) <= max_seq_length]

        self.ids = ids
        self.test_size = test_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train, self.val = tts(self.ids, test_size=self.test_size)

    def train_dataloader(self):
        return DataLoader(
            DummyDataset(self.train),
            batch_sampler=DummySampler(
                SequentialSampler(sorted(range(len(self.train)), key=lambda x: len(self.train[x][0]))),
                batch_size=self.batch_size, 
                drop_last=False, 
                ids=self.train
            ),
            collate_fn=self.collate_fn,
            num_workers=8
        )
    
    def val_dataloader(self):
        return DataLoader(
            DummyDataset(self.val),
            batch_sampler=DummySampler(
                SequentialSampler(sorted(range(len(self.val)), key=lambda x: len(self.val[x][0]))),
                batch_size=self.batch_size, 
                drop_last=False, 
                ids=self.val
            ),
            collate_fn=self.collate_fn,
            num_workers=8
        )
    
    def collate_fn(self, batch):
        
        # src, src_key_padding_mask, tgt, tgt_key_padding_mask

        source_input_ids = np.zeros((len(batch), max(map(lambda x: len(x[0]), batch))))
        target_input_ids = np.zeros((len(batch), max(map(lambda x: len(x[1]), batch))))
        source_mask= np.zeros_like(source_input_ids, dtype=bool)
        target_mask= np.zeros_like(target_input_ids, dtype=bool)

        for i, (source, target) in enumerate(batch):
            source_input_ids[i, :len(source)] = source
            target_input_ids[i, :len(target)] = target
            source_mask[i, len(source):] = True
            target_mask[i, len(target):] = True
        
        return torch.from_numpy(source_input_ids).long(), \
            torch.from_numpy(source_mask), \
            torch.from_numpy(target_input_ids).long(), \
            torch.from_numpy(target_mask), \
            np.count_nonzero(source_input_ids) / source_input_ids.size


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


if __name__ == '__main__':

    from pytorch_lightning.utilities.cli import LightningCLI

    cli = LightningCLI(Translator, TranslationDataModule)