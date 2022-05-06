# coding: UTF-8
import torch
from torch.utils.data import Dataset
from typing import List
from script import Sentence, Label
import numpy as np

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class BiLSTMDataset(Dataset):
    def __init__(self, config, sentences: List[Sentence], labels: List[Label]):
        self.config = config
        self.dataset = self.preprocess(sentences, labels)
        self.label2id = config.label2id

    def preprocess(self, sentences: List[Sentence], labels: List[Label]):
        """convert the data to ids"""
        processed = []
        cnt = 0
        for k, (sentence, label) in enumerate(zip(sentences, labels)):
            try:
                word_id = self._convert_sentence2id(sentence)
                label_id = self._convert_label2id(label, len(word_id))
                processed.append((word_id, label_id))
            except Exception as e:
                cnt = cnt+1
                self.config.logger.warning("preprocess data: index:{}, cnt:{}, error:{}".format(k, cnt, str(e)))
        self.config.logger.info("-------- Process Done! --------")
        return processed

    def _convert_label2id(self, label: Label, label_len: int):
        true_label = ["O"] * label_len
        label = label.get_label()
        for entity, name in label.items():
            for word, indexs in name.items():
                for index in indexs:
                    for _, start in enumerate(range(index[0], index[1] + 1)):
                        if _ == 0:
                            true_label[start] = "B-" + entity
                        else:
                            true_label[start] = "I-" + entity

        label_id = [self.config.label2id[l_] for l_ in true_label]
        return label_id

    def _convert_sentence2id(self, sentence: Sentence):
        sentence = sentence.get_text()
        word_id = [self.config.word2index.get(word, self.config.word2index[UNK]) for word in sentence]
        return word_id

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, texts, labels, batch_size):

        token_len = max([len(x) for x in texts])
        label_len = max([len(x) for x in labels])
        text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, label_len).fill_(self.config.label2id["O"])
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(texts, labels)):
            text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

        return text_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):

        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in labels]
        batch_size = len(batch)

        input_ids, label_ids, input_mask = self.get_long_tensor(texts, labels, batch_size)

        return (input_ids, input_mask), lens, label_ids


class BertDataset(BiLSTMDataset):

    def preprocess(self, sentences: List[Sentence], labels: List[Label]):
        """convert the data to ids"""
        processed = []
        cnt = 0
        for k, (sentence, label) in enumerate(zip(sentences, labels)):
            try:
                word_id = self._convert_sentence2id(sentence)
                label_id = self._convert_label2id(label, len(word_id)-1)
                # label_id = [self.config.label2id["O"]] + label_id
                processed.append((word_id, label_id))
            except Exception as e:
                cnt = cnt + 1
                self.config.logger.warning("preprocess data: index:{}, cnt:{}, error:{}".format(k, cnt, str(e)))
        self.config.logger.info("-------- Process Done! --------")

        return processed

    def _convert_sentence2id(self, sentence: Sentence):
        words = []
        for token in sentence.get_text():
            token = self.config.tokenizer.tokenize(token)
            token = token if token else ["[SEP]"]
            words.append(token)
        # 变成单个字的列表，开头加上[CLS]
        words = ['[CLS]'] + [item for token in words for item in token]
        word_id = self.config.tokenizer.convert_tokens_to_ids(words)
        return word_id


class BertGlobalPointerDataset(Dataset):
    def __init__(self, config, sentences: List[Sentence], labels: List[Label]):
        self.config = config
        self.sentences = sentences
        self.labels = labels
        self.label2id = config.label2id

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return [sentence, label]

    def __len__(self):
        return len(self.sentences)

    def _convert_label2id(self, label: Label, max_seq_len: int):
        label_id = np.zeros((self.config.entity_classes, max_seq_len, max_seq_len))

        for entity, name in label.get_label().items():
            for _, index_list in name.items():
                for start, end in index_list:
                    label_id[self.config.entity2id[entity], start, end] = 1
        return label_id

    def _convert_sentence2id(self, sentence: Sentence):
        words = []
        for token in sentence.get_text():
            token = self.config.tokenizer.tokenize(token)
            token = token if token else ["[SEP]"]
            words.append(token)
        words = [item for token in words for item in token]
        word_id = self.config.tokenizer.convert_tokens_to_ids(words)
        return word_id

    def _get_long_tensor_sentence(self, sentences: List[Sentence]):
        token_ids = [self._convert_sentence2id(i) for i in sentences]
        max_seq_len = max([len(x) for x in token_ids])
        batch_size = len(token_ids)

        text_tokens = torch.LongTensor(batch_size, max_seq_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, max_seq_len).fill_(0)

        for i, s in enumerate(token_ids):
            text_tokens[i, :len(s)] = torch.LongTensor(s)
            mask_tokens[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.uint8)
        return text_tokens, mask_tokens, max_seq_len

    def _get_long_tensor_label(self, labels: List[Label], max_seq_len: int):
        labels_list = [self._convert_label2id(i, max_seq_len) for i in labels]
        labels_list = [torch.tensor(i).long() for i in labels_list]
        batch_labels = torch.stack(labels_list, dim=0)
        return batch_labels

    def collate_fn(self, batch):

        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x.get_text()) for x in sentences]

        input_ids, input_mask, max_seq_len = self._get_long_tensor_sentence(sentences)
        label_ids = self._get_long_tensor_label(labels, max_seq_len)

        return (input_ids, input_mask), lens, label_ids





