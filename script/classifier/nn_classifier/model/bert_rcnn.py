# coding: UTF-8
from .bert import BertClassifier
from .bert import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class RCNNConfig(Config):
    """配置参数"""

    def __init__(self, **kwargs):
        Config.__init__(self,
                        dataset=kwargs.get('dataset'),
                        cache_dir=kwargs.get('cache_dir'),
                        model_name=kwargs.get('model_name'),
                        adversarial=kwargs.get('adversarial'),)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x, labels=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)

        if labels is not None:
            # loss_mask = labels.gt(-1)
            loss = F.cross_entropy(out, labels)
            return out, loss

        return out


class BertRCNNClassifier(BertClassifier, RCNNConfig):
    def __init__(self, **kwargs):
        RCNNConfig.__init__(self,
                           dataset=kwargs.get('dataset'),
                           cache_dir=kwargs.get('cache_dir'),
                           model_name=kwargs.get('model_name'))
        self.model = Model(self).to(self.device)
