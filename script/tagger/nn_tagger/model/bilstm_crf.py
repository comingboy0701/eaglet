# coding: UTF-8
import os
import torch
import torch.nn as nn
from ..crf import CRF
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from script.pretraining import load_emb_vec
from script.pretraining import Vocabulary
from importlib import import_module
from script.tagger.nn_tagger.tagger import Tagger
from script.tagger.nn_tagger.text_utils import BiLSTMDataset
from script.tagger.setting import DataConfig


class Config(DataConfig):
    """数据配置参数"""

    def __init__(self, **kwargs):
        DataConfig.__init__(self,
                            dataset=kwargs.get('dataset'),
                            cache_dir=kwargs.get('cache_dir'),
                            model_name=kwargs.get('model_name'),
                            adversarial=kwargs.get('adversarial'),
                            )
        if self.emb_path and os.path.exists(self.emb_path):
            init_embedding, self.word2index = load_emb_vec(self.emb_path)
            self.embedding_pretrained = torch.tensor(init_embedding.astype('float32'))
            self.embedding_size = self.embedding_pretrained.size(1)
        else:
            self.embedding_pretrained = None
            self.embedding_size = 128
            vocabulary = Vocabulary(train_files=[self.train_path, self.test_path, self.dev_path],
                                    emb_path=self.emb_path,
                                    vocab_path=self.vocab_path,
                                    max_vocab_size=100000)
            self.word2index = vocabulary.get_vocab()

        self.n_vocab = len(self.word2index) + 1  # 词表大小，在运行时赋值
        self.n_split = 5
        self.dev_split_size = 0.1
        self.batch_size = 32
        self.hidden_size = 384
        self.dropout = 0.5  # 随机失活
        self.learning_rate = 0.001  # 学习率
        self.betas = (0.9, 0.999)
        self.lr_step = 5
        self.lr_gamma = 0.8
        self.num_epochs = 30
        self.require_improvement = 1000  # 若超过100batch效果还没提升，则提前结束训练


class BiLstmCrf(nn.Module):

    def __init__(self, config):
        super(BiLstmCrf, self).__init__()
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # self.embedding = nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=config.n_vocab - 1)
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_size)

        self.bilstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=config.dropout,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_classes)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, x, labels=None):
        input_ids, input_mask = x
        embeddings = self.embedding(input_ids)
        sequence_output, _ = self.bilstm(embeddings)
        tag_scores = self.classifier(sequence_output)
        if labels is not None:
            loss = self.crf(tag_scores, labels, input_mask) * (-1)
            return tag_scores, loss
        return tag_scores

    def crf_decode(self, tag_scores, mask):
        labels_pred = self.crf.decode(tag_scores, mask=mask)
        return labels_pred


class BiLstmCrfTagger(Tagger, Config):
    def __init__(self, **kwargs):
        Config.__init__(self, **kwargs)
        self.model = BiLstmCrf(self).to(self.device)
        self.logger.info(self.model.parameters)
        adversarial = kwargs.get('adversarial', 'base')
        at_model = import_module("script.adversarial.{}_model".format(adversarial))
        self.atModel = at_model.ATModel(self.model)
        self.modelDataset = BiLSTMDataset
        self._set_optimizer()
        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_state_dict(torch.load(self.save_model_path))
            self.model.eval()
            self.logger.info("load {} success".format(self.model_name))

    def _set_optimizer(self, ):
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
