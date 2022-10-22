import torch
from torch.utils.data import Dataset
from typing import List

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class BertDataset(Dataset):
    def __init__(self, config, data: List[dict]):
        self.config = config
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        sentences = []
        y = []
        for line in batch:
            text_a, label = line["text_a"], line["label"]
            if "text_b" in line:
                text_b = line["text_b"]
                sentences.append([text_a, text_b])
            else:
                sentences.append(text_a)

            y.append(self.config.label2id.get(label, 0))
        x = self.config.tokenizer(sentences, max_length=self.config.pad_size, truncation=True, return_tensors="pt",
                                  padding=True)
        y = torch.tensor(y, dtype=torch.long)
        return (x["token_id"], x[""], ["mask"]), y


class TextCnnDataset(Dataset):
    def __init__(self, config, data: List[dict]):
        self.config = config
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        tokens, seq_len, y, seq_len = [], [], [], []
        for line in batch:
            text_a, label = line["text_a"], line["label"]
            text_token = [self.config.word2index.get(word, self.config.word2index[UNK]) for word in text_a]
            if "text_b" in line:
                text_b = line["text_b"]
                text_b_token = [self.config.word2index.get(word, self.config.word2index[UNK]) for word in text_b]
                text_token = text_token+[self.config.word2index[PAD]] + text_b_token
            tokens.append(text_token)
            seq_len.append(len(tokens))
            y.append(self.config.label2id.get(label, 0))

        seq_len = [min(i, self.config.pad_size) for i in seq_len]
        max_len = max(seq_len)

        tokens_pad = []
        for tokens in tokens:
            if len(tokens) >= max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.config.word2index[PAD]] * (max_len - len(tokens))
            tokens_pad.append(tokens)

        feature = torch.tensor(tokens_pad, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return (feature, seq_len), y


