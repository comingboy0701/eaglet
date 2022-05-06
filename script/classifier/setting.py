import multiprocessing as P
import os
from script.utils.file import load_json
from script import get_logger, limit_gpu_memory
import torch
from setting import ROOT_PATH


class DataConfig(object):
    """数据配置参数"""

    def __init__(self, dataset=None, cache_dir=None, model_name=None, adversarial=None):
        if dataset:
            self.train_path = os.path.join(dataset, 'data', 'train.json')  # 训练集
            self.dev_path = os.path.join(dataset, 'data', 'dev.json')  # 验证集
            self.test_path = os.path.join(dataset, 'data', 'test.json')  # 测试集
            self.label2id = load_json(os.path.join(dataset, 'data', 'label2id.json'))  # 类别名单
            self.id2label = {j: i for i, j in self.label2id.items()}
            self.pred_path = os.path.join(dataset, 'data', 'predict.csv')  # 预测集 必须有一列sentence
            self.save_pred_path = os.path.join(dataset, 'cache', 'predict_cls_result.csv')  # 预测集结果
            self.num_classes = len(self.label2id)  # 类别数
            self.topk = 3 if self.num_classes > 3 else self.num_classes  # topk
            self.random = True

        if (not cache_dir) and dataset:
            cache_dir = os.path.join(dataset, 'cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_name = model_name
        self.logger = get_logger(name=model_name + "_" + adversarial, log_dir=self.cache_dir)
        if model_name.startswith("nn"):
            self.save_model_path = os.path.join(self.cache_dir, model_name + "_" + adversarial + '.ckpt')  # 模型训练结果
        elif model_name.startswith("tf"):
            self.save_model_path = os.path.join(self.cache_dir, model_name + "_" + adversarial + '.weight')  # 模型训练结果

        self.emb_path = os.path.join(ROOT_PATH, "embedding", "char-SougouNews.vec")
        self.vocab_path = os.path.join(cache_dir, 'vocab.json')  # 自动生成word2id的json文件
        self.device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu')  # 设备
        self.cpu_count = min(P.cpu_count(), 16)
        self.toy = False
        self.memory_limit = 1024 * 8
        self.gpu_no = 1
        limit_gpu_memory(self.memory_limit, self.gpu_no)

