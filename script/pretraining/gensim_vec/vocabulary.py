# -*- coding: utf-8 -*-
import os
import logging
from script.utils import load_json, save_json
from script import Sentence
from .tokenizer import Tokenizer
from .load_emb_vec import load_emb_vec
from typing import List


class Vocabulary:
    """
    构建词表
    """

    def __init__(self,
                 train_files: List[str] = None,
                 emb_path: str = None,
                 vocab_path: str = None,
                 max_vocab_size=10000):
        self.train_files = train_files
        self.emb_path = emb_path
        self.vocab_path = vocab_path
        self.max_vocab_size = max_vocab_size
        self.word2id = {}
        self.id2word = None

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    # 获取词的id
    def word_id(self, word):
        return self.word2id[word]

    # 获取id对应的词
    def id_word(self, idx):
        return self.id2word[idx]

    def get_vocab(self):
        """
        进一步处理，将word和label转化为id
        word2id: dict,每个字对应的序号
        idx2word: dict,每个序号对应的字
        保存为二进制文件
        """
        # 如果有处理好的，就直接load
        if self.emb_path and os.path.exists(self.emb_path):
            _, word2index = load_emb_vec(self.emb_path)
            return word2index

        if self.vocab_path and os.path.exists(self.vocab_path):
            word2index = load_json(self.vocab_path)
            return word2index

        # 如果没有处理训练和测试文件
        tokenizer = Tokenizer(num_words=self.max_vocab_size,
                              filters='',
                              lower=True,
                              split='',
                              char_level=True,
                              oov_token=None,
                              document_count=0
                              )

        for file in self.train_files:
            if file and os.path.exists(file):
                sentences = Sentence.load(file)
                tokenizer.fit_on_texts([i.get_text() for i in sentences])

        word_index = tokenizer.word_index
        word2id = {word: index for index, word in enumerate(word_index) if index < self.max_vocab_size}

        if "<PAD>" not in word2id:
            word2id["<PAD>"] = len(word2id)
        if "<UNK>" not in word2id:
            word2id["<UNK>"] = len(word2id)
        if " " not in word2id:
            word2id[" "] = len(word2id)

        def swap(word2id, id2word, word, position=0):
            if word2id[word] == position:
                return word2id, id2word
            else:
                swap_word = id2word[position]
                swap_position = word2id[word]
                word2id[word], word2id[swap_word] = word2id[swap_word], word2id[word]
                id2word[position], id2word[swap_position] = id2word[swap_position], id2word[position]
            return word2id, id2word

        id2word = {index: word for word, index in word2id.items()}
        word2id, id2word = swap(word2id, id2word, '<PAD>', position=0)
        word2id, id2word = swap(word2id, id2word, '<UNK>', position=1)
        word2id, id2word = swap(word2id, id2word, ' ', position=2)

        word2id = dict(sorted(word2id.items(), key=lambda x: x[1], reverse=False))
        # 保存为word2id文件
        save_json(word2id, self.vocab_path)
        return word2id


if __name__ == '__main__':
    files = ["/dataset/实体提取/cluener_public/data/train.json",
             "/dataset/实体提取/cluener_public/data/dev.json",
             "/dataset/实体提取/cluener_public/data/test.json"
             ]
    emb_path = "/embedding/char-SougouNews.vec"
    vocab_path = "dataset/实体提取/cluener_public/cache/vocab.json"
    max_vocab_size = 10000
    vocab = Vocabulary(files, emb_path, vocab_path, max_vocab_size)
    word2id = vocab.get_vocab()
