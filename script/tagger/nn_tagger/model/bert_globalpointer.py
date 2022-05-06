# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertTokenizer
from importlib import import_module
from script.tagger.nn_tagger.tagger_span import TaggerSpan
from script.tagger.nn_tagger.text_utils import BertGlobalPointerDataset
from script.tagger.setting import DataConfig
import sys

from script.tagger.nn_tagger.loss import multilabel_categorical_crossentropy


class Config(DataConfig):
    """数据配置参数"""

    def __init__(self, **kwargs):
        DataConfig.__init__(self,
                            dataset=kwargs.get('dataset'),
                            cache_dir=kwargs.get('cache_dir'),
                            model_name=kwargs.get('model_name'),
                            adversarial=kwargs.get('adversarial'), )
        self.require_improvement = 5000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_epochs = 30  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.learning_rate = 5e-5  # 学习率
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.weight_decay = 0.01
        self.clip_grad = 5
        self.dev_split_size = 0.1
        self.RoPE = True
        self.inner_dim = 64

        # self.bert_type = "bert"
        # self.bert_path = 'bert_model/NN/bert-base-chinese'
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.bert_type = "nezha"
        self.bert_path = 'bert_model/NN/nezha-chinese-base'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)


class BertGlobalPointer(nn.Module):
    def __init__(self, config):
        super(BertGlobalPointer, self).__init__()
        self.config = config
        self.inner_dim = config.inner_dim
        self.device = config.device
        self.RoPE = config.RoPE
        self.entity_classes = config.entity_classes
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
        self.dense = nn.Linear(config.hidden_size, config.entity_classes * config.inner_dim * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.config.device)
        return embeddings

    def forward(self, x, labels=None):
        input_ids, attention_mask = x
        if self.config.bert_type == "bert":
            context_outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=False,
                                           return_dict=False)
        elif self.config.bert_type == "nezha":
            context_outputs= self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        else:
            sys.exit("config 必须提供 bert_type")

        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, entity_classes*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, entity_classes, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, entity_classes, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, entity_classes, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.entity_classes, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.entity_classes, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits / self.inner_dim ** 0.5

        if labels is not None:
            loss = self.loss_fun(logits, labels)
            return logits, loss

        return logits

    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss


class BertGlobalPointerTagger(TaggerSpan, Config):
    def __init__(self, **kwargs):
        Config.__init__(self, **kwargs)
        self.model = BertGlobalPointer(self).to(self.device)
        # self.logger.info(self.model.parameters)
        adversarial = kwargs.get('adversarial', 'base')
        at_model = import_module("script.adversarial.{}_model".format(adversarial))
        self.atModel = at_model.ATModel(self.model)
        self.modelDataset = BertGlobalPointerDataset
        self._set_optimizer()
        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_state_dict(torch.load(self.save_model_path))
            self.model.eval()
            self.logger.info("load {} success".format(self.model_name))

    def _set_optimizer(self, ):

        # bert_optimizer = list(self.model.bert.named_parameters())
        # classifier_optimizer = list(self.model.classifier.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
        #      'weight_decay': self.weight_decay},
        #     {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0},
        #     {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
        #      'lr': self.learning_rate * 5, 'weight_decay': self.weight_decay},
        #     {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
        #      'lr': self.learning_rate * 5, 'weight_decay': 0.0},
        #     {'params': self.model.crf.parameters(), 'lr': self.learning_rate * 1000}
        # ]
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
        # self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=(self.num_epochs // 10) * train_per_epoch,
                                                         num_training_steps=self.num_epochs * train_per_epoch
                                                         )
