# coding: UTF-8
from .bert import BertClassifier
from .bert import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class CNNConfig(Config):
    """配置参数"""

    def __init__(self, **kwargs):
        Config.__init__(self,
                        dataset=kwargs.get('dataset'),
                        cache_dir=kwargs.get('cache_dir'),
                        model_name=kwargs.get('model_name'),
                        adversarial=kwargs.get('adversarial'),)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, labels=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_hidden_states=False, return_dict=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        if labels is not None:
            # loss_mask = labels.gt(-1)
            loss = F.cross_entropy(out, labels)
            return out, loss
        return out


class BertCNNClassifier(BertClassifier, CNNConfig):
    def __init__(self, **kwargs):
        CNNConfig.__init__(self,
                           dataset=kwargs.get('dataset'),
                           cache_dir=kwargs.get('cache_dir'),
                           model_name=kwargs.get('model_name'))
        self.model = Model(self).to(self.device)
