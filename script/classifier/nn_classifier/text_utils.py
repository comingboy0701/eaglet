import torch
from torch.utils.data import Dataset
from typing import List
from script import Sentence, Label

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class BertDataset(Dataset):
    def __init__(self, config, sentences: List[Sentence], labels: List[Label]):
        self.config = config
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence, label = self.sentences[index], self.labels[index]

        tokenizer = self.config.tokenizer(sentence.get_text(), max_length=self.config.pad_size,
                                          truncation=True, return_tensors="pt", padding='max_length')
        x = tokenizer["input_ids"][0]
        mask = tokenizer["attention_mask"][0]
        seq_len = torch.sum(mask)
        y = torch.tensor(int(label.get_label()), dtype=torch.long)
        return (x, seq_len, mask), y


class TextCnnDataset(Dataset):
    def __init__(self, config, sentences: List[Sentence], labels: List[Label]):
        self.config = config
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence, label = self.sentences[index], self.labels[index]
        y = torch.tensor(int(label.get_label()), dtype=torch.long)

        sentence = sentence.get_text()
        tokens = [self.config.word2index.get(word, self.config.word2index[UNK]) for word in sentence]
        seq_len = torch.tensor(min(self.config.pad_size, len(tokens)), dtype=torch.long)

        tokens = self.pad(tokens)
        feature = torch.tensor(tokens, dtype=torch.long)

        return (feature, seq_len), y

    def pad(self, tokens):
        if len(tokens) >= self.config.pad_size:
            return tokens[:self.config.pad_size]
        else:
            return tokens + [self.config.word2index[PAD]] * (self.config.pad_size - len(tokens))

