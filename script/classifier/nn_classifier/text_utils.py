import torch
from torch.utils.data import Dataset
from typing import List
from script import Sentence, Label

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
            text_a, label = line["text"], line["label"]
            if "text_b" in line:
                text_b = line["text_b"]
                sentences.append([text_a, text_b])
            else:
                sentences.append(text_a)

            y.append(self.config.label2id.get(label, 0))
        x = self.config.tokenizer(sentences, max_length=self.config.pad_size, truncation=True,
                                  return_tensors="pt", padding=True)
        y = torch.tensor(y, dtype=torch.long)

        return (x["input_ids"], x["token_type_ids"], x["attention_mask"]), y


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

