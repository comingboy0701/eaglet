# coding: UTF-8
import os
import torch
import torch.nn as nn
from ..crf import CRF
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from .optimization import BertAdam
from transformers import BertTokenizer
from importlib import import_module
from script.tagger.nn_tagger.tagger import Tagger
from script.tagger.nn_tagger.text_utils import BertDataset
from script.tagger.setting import DataConfig
import sys


class Config(DataConfig):
    """数据配置参数"""

    def __init__(self, **kwargs):
        DataConfig.__init__(self,
                            dataset=kwargs.get('dataset'),
                            cache_dir=kwargs.get('cache_dir'),
                            model_name=kwargs.get('model_name'),
                            adversarial=kwargs.get('adversarial'), )
        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 30  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.learning_rate = 5e-5  # 学习率
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.weight_decay = 0.01
        self.clip_grad = 5
        self.dev_split_size = 0.1

        # self.bert_type = "bert"
        # self.bert_path = 'bert_model/NN/bert-base-chinese'
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.bert_type = "nezha"
        self.bert_path = 'bert_model/NN/nezha-chinese-base'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)


class BertCrf(nn.Module):

    def __init__(self, config):
        super(BertCrf, self).__init__()
        self.config = config
        if config.bert_type == "bert":
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(config.bert_path)
        elif config.bert_type == "nezha":
            from script.model.NEZHA.modeling_nezha import BertModel
            self.bert = BertModel.from_pretrained(config.bert_path)
        else:
            sys.exit("config 必须提供 bert_type")
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, x, labels=None):
        input_ids, input_mask = x
        if self.config.bert_type == "bert":
            sequence_output, _ = self.bert(input_ids, attention_mask=input_mask, output_hidden_states=False, return_dict=False)
        elif self.config.bert_type == "nezha":
            sequence_output, _ = self.bert(input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        else:
             sys.exit("config 必须提供 bert_type")
        # # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        # origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
        #                           for layer, starts in zip(sequence_output, input_token_starts)]
        # # 将sequence_output的pred_label维度padding到最大长度
        # padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # # dropout pred_label的一部分feature
        # sequence_output = self.dropout(sequence_output)
        # 得到判别值
        sequence_output = sequence_output[:, 1:, :]  # 去除【CLS】标签位置，获得与label对齐的pre_label表示
        input_mask = input_mask[:, 1:]
        tag_scores = self.classifier(sequence_output)
        if labels is not None:
            # loss_mask = labels.gt(-1)
            loss = self.crf(tag_scores, labels, input_mask) * (-1)
            return tag_scores, loss

        return tag_scores
    
    def crf_decode(self, tag_scores, mask):
        labels_pred = self.crf.decode(tag_scores, mask=mask[:, 1:])
        return labels_pred


class BertCrfTagger(Tagger, Config):
    def __init__(self, **kwargs):
        Config.__init__(self, **kwargs)
        self.model = BertCrf(self).to(self.device)
        # self.logger.info(self.model.parameters)
        adversarial = kwargs.get('adversarial', 'base')
        at_model = import_module("script.adversarial.{}_model".format(adversarial))
        self.atModel = at_model.ATModel(self.model)
        self.modelDataset = BertDataset
        self._set_optimizer()
        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_state_dict(torch.load(self.save_model_path))
            self.model.eval()
            self.logger.info("load {} success".format(self.model_name))

    def _set_optimizer(self, ):

        bert_optimizer = list(self.model.bert.named_parameters())
        classifier_optimizer = list(self.model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': self.learning_rate * 5, 'weight_decay': 0.0},
            {'params': self.model.crf.parameters(), 'lr': self.learning_rate * 1000}
        ]
        train_iter = self._covert_sentence_label(self.train_path, self.toy)
        if self.dev_path and os.path.exists(self.dev_path):  # exists dev_path, don't split train data
            train_per_epoch = len(train_iter) // self.batch_size
        else:
            train_per_epoch = len(train_iter) * (1 - self.dev_split_size) // self.batch_size
        # self.optimizer = BertAdam(optimizer_grouped_parameters,
        #                           lr=self.learning_rate,
        #                           warmup=0.05,
        #                           t_total=len(train_iter) * self.num_epochs)
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, correct_bias=False)
        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=(self.num_epochs // 10) * train_per_epoch,
                                                         num_training_steps=self.num_epochs * train_per_epoch
                                                         )
