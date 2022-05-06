# coding: UTF-8
import os
import torch
import torch.nn as nn
from script.classifier.nn_classifier import Classifier
from script.classifier.nn_classifier.text_utils import TextCnnDataset
from script.classifier.setting import DataConfig
import torch.nn.functional as F
from script.pretraining import load_emb_vec
from importlib import import_module
from script.pretraining import Vocabulary


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
            self.embed = self.embedding_pretrained.size(1)
        else:
            self.embedding_pretrained = None
            self.embed = 300
            vocabulary = Vocabulary(train_files=[self.train_path, self.test_path, self.dev_path],
                                    emb_path=self.emb_path,
                                    vocab_path=self.vocab_path,
                                    max_vocab_size=10000)
            self.word2index = vocabulary.get_vocab()
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.n_vocab = len(self.word2index)+1  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, labels=None):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        if labels is not None:
            # loss_mask = labels.gt(-1)
            loss = F.cross_entropy(out, labels)
            return out, loss

        return out


class TextCNNClassifier(Classifier, Config):
    def __init__(self, **kwargs):
        Config.__init__(self, **kwargs)
        self.model = Model(self).to(self.device)
        self.logger.info(self.model.parameters)
        adversarial = kwargs.get('adversarial', 'base')
        at_model = import_module("script.adversarial.{}_model".format(adversarial))
        self.atModel = at_model.ATModel(self.model)
        self.modelDataset = TextCnnDataset
        self._set_optimizer()
        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_state_dict(torch.load(self.save_model_path))
            self.model.eval()
            self.logger.info("load {} success".format(self.model_name))

    def _set_optimizer(self, ):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
