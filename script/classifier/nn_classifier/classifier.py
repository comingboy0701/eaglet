import numpy as np
import pandas as pd
import torch
import time
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from script.utils.utils import get_time_dif
from script import Sentence, Label
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, *args, **kwargs):
        self.model = None
        self.logger = None
        self.batch_size = None
        self.device = None
        self.train_path = None
        self.dev_path = None
        self.test_path = None
        self.save_model_path = None
        self.label2id = None
        self.id2label = None
        self.pred_path = None
        self.save_pred_path = None
        self.num_classes = None
        self.num_epochs = None
        self.topk = None
        self.require_improvement = None
        self.cpu_count = None
        self.modelDataset = None
        self.toy = None
        self.optimizer = None
        self.random = None
        self.loss_fun = None
        self.optimizer = None
        self.atModel = None

    def persist_model(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load_model(cls, path: str) -> "Classifier":

        raise NotImplementedError()

    @staticmethod
    def _sentence_load(path):
        return Sentence.load(path)

    @staticmethod
    def _label_load(path):
        return Label.load(path)

    def _covert_sentence_label(self, path: str, toy: bool = False):
        sentences, labels = self._sentence_load(path), self._label_load(path)
        if toy:
            _, sentences, _, labels = train_test_split(sentences, labels, test_size=0.1)
        ds_data = self.modelDataset(self, sentences, labels)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size, shuffle=self.random, num_workers=self.cpu_count)
        return data_iter

    def train_model(self, ) -> None:

        train_iter = self._covert_sentence_label(self.train_path, self.toy)
        dev_iter = self._covert_sentence_label(self.dev_path, self.toy)

        start_time = time.time()
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        self.model.train()
        for epoch in range(self.num_epochs):
            self.logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                trains = [i.to(self.device) for i in trains]
                labels = labels.to(self.device)
                outputs, loss = self.atModel.train(trains, labels, self.optimizer)
                # outputs = self.model(trains)
                # self.model.zero_grad()
                # loss = F.cross_entropy(outputs, labels)
                # loss.backward()
                # self.optimizer.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    self.model.eval()
                    true = labels.data.cpu()
                    predict = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predict)
                    dev_acc, dev_loss = self._evaluate(self.model, dev_iter, train=True)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(self.model.state_dict(), self.save_model_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    self.logger.info(
                        msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    self.model.train()
                total_batch += 1
                if total_batch - last_improve > self.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    self.logger.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        time_dif = get_time_dif(start_time)
        self.logger.info("Train time usage:{}".format(time_dif))

    def evaluate_model(self, ):
        start_time = time.time()
        test_iter = self._covert_sentence_label(self.test_path, toy=False)
        test_acc, test_loss, test_report, test_confusion = self._evaluate(self.model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        self.logger.info(msg.format(test_loss, test_acc))
        self.logger.info("Precision, Recall and F1-Score...")
        self.logger.info(test_report)
        self.logger.info("Confusion Matrix...")
        self.logger.info(test_confusion)
        time_dif = get_time_dif(start_time)
        self.logger.info("Test time usage:{}".format(time_dif))

    def _evaluate(self, model, data_iter, train=False, test=False, predict=False):

        loss_total = 0
        true_labels = []
        pred_labels = []
        pred_scores = []
        with torch.no_grad():
            for texts, labels in data_iter:
                texts = [i.to(self.device) for i in texts]
                outputs = model(texts)

                pred_label = torch.topk(outputs.data, self.topk, dim=1, largest=True, sorted=True, out=None)[
                    1].cpu().numpy()
                pred_score = \
                torch.topk(torch.nn.functional.softmax(outputs.data, 1), self.topk, dim=1, largest=True, sorted=True,
                           out=None)[0].cpu().numpy()
                pred_labels.append(pred_label)
                pred_scores.append(pred_score)

                if train or test:
                    true_labels.append(labels)
                    labels = labels.to(self.device)
                    loss = F.cross_entropy(outputs, labels)
                    loss_total += loss
        if predict:
            return np.concatenate(pred_labels).astype("int8"), np.concatenate(pred_scores).astype("float16")

        true_labels = np.concatenate(true_labels)
        pred_labels = np.concatenate(pred_labels)[:, 0]
        acc = metrics.accuracy_score(true_labels, pred_labels)

        if test:
            report = metrics.classification_report(true_labels, pred_labels, target_names=list(self.label2id),
                                                   digits=4)
            confusion = metrics.confusion_matrix(true_labels, pred_labels)
            return acc, loss_total / len(data_iter), report, confusion

        if train:
            return acc, loss_total / len(data_iter)

    def predict_offline(self, ):

        df_pred = pd.read_csv(self.pred_path)
        sentences = [Sentence(i) for i in df_pred["sentence"]]
        labels = [Label("0") for _ in df_pred["sentence"]]
        ds_data = self.modelDataset(self, sentences, labels)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpu_count)

        pred_labels, pred_scores = self._evaluate(self.model, data_iter, predict=True)

        df_label = pd.DataFrame(data=pred_labels, columns=["top_k{}".format(i) for i in range(1, self.topk + 1)])
        df_scores = pd.DataFrame(data=pred_scores, columns=["top_k{}_prob".format(i) for i in range(1, self.topk + 1)])
        df_result = pd.concat([df_pred, df_label, df_scores], axis=1)
        df_result.to_csv(self.save_pred_path, index=False, float_format="%.4f", sep="\t", header=True)
        self.logger.info("predict success...")

    def predict_online(self, sentences_list: List[dict]):
        """传入参数
        # [{
        #     "unique_id": "1234",
        #     "text_a": '开开心心',
        #     "text_b": '每一天',
        # }]
        """
        self.logger.info("预测数据: {}".format(sentences_list))
        data_list = []
        for i in sentences_list:
            i["label"] = '0'
            data_list.append(i)
        ds_data = self.modelDataset(self, data_list)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpu_count)
        pred_labels, pred_scores = self._evaluate(self.model, data_iter, predict=True)
        for i, j, k in zip(sentences_list, pred_labels, pred_scores):
            i.update({"pred_label_id": str(j[0]), 'pred_label': self.id2label[j[0]], "pred_score": str(round(k[0], 4))})
        return