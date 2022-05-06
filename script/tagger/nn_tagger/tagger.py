# coding: UTF-8
import os
import numpy as np
import pandas as pd
import torch
import time
from sklearn import metrics
from torch.utils.data import DataLoader
from script.utils.utils import get_time_dif
from script import Sentence, Label
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
from ..metric import f1_score, get_entities
import sys


class Tagger:

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
        self.entity2id = None
        self.id2entity = None
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
        self.dev_split_size = None
        self.scheduler = None

    def persist_model(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load_model(cls, path: str) -> "Tagger":

        raise NotImplementedError()

    @staticmethod
    def _sentence_load(path):
        return Sentence.load(path)

    @staticmethod
    def _label_load(path):
        return Label.load(path)

    def _covert_sentence_label(self, path: str, toy: bool = False, shuffle: bool = True):
        sentences, labels = self._sentence_load(path), self._label_load(path)
        if toy:
            _, sentences, _, labels = train_test_split(sentences, labels, test_size=0.1)
        ds_data = self.modelDataset(self, sentences, labels)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.cpu_count, collate_fn=ds_data.collate_fn)
        return data_iter

    def _train_test_split(self, path: str):

        train_sentences, train_labels = self._sentence_load(path), self._label_load(path)
        train_sentences, test_sentences, train_labels, test_labels = \
            train_test_split(train_sentences, train_labels, test_size=self.dev_split_size, random_state=666)

        ds_data = self.modelDataset(self, train_sentences, train_labels)
        train_iter = DataLoader(ds_data, batch_size=self.batch_size,
                                shuffle=self.random, num_workers=self.cpu_count, collate_fn=ds_data.collate_fn)

        ds_data = self.modelDataset(self, test_sentences, test_labels)
        dev_iter = DataLoader(ds_data, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.cpu_count, collate_fn=ds_data.collate_fn)
        return train_iter, dev_iter

    def train_model(self, ) -> None:
        if self.dev_path and os.path.exists(self.dev_path):
            train_iter = self._covert_sentence_label(self.train_path, self.toy, self.random)
            dev_iter = self._covert_sentence_label(self.dev_path, self.toy, False)
        else:
            train_iter, dev_iter = self._train_test_split(self.train_path)
        self.logger.info("train数据: {}".format(len(train_iter.dataset)))
        self.logger.info("dev数据: {}".format(len(dev_iter.dataset)))

        start_time = time.time()
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        dev_best_f1_score = 0
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        self.model.train()
        for epoch in range(self.num_epochs):
            self.logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for i, (trains, x_lens, labels) in enumerate(train_iter):
                trains = [i.to(self.device) for i in trains]
                labels = labels.to(self.device)
                tag_scores, loss = self.atModel.train(trains, labels, self.optimizer)
                self.optimizer.zero_grad()

                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    self.model.eval()

                    labels_true = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), x_lens)]
                    # labels_pred = self.model.crf.decode(tag_scores, mask=trains[1])
                    labels_pred = self.model.crf_decode(tag_scores, mask=trains[1])

                    # 标签
                    train_tag_acc = metrics.accuracy_score(np.hstack(labels_true), np.hstack(labels_pred))

                    # 实体
                    true_tag = [[self.id2label[j] for j in i] for i in labels_true]
                    pred_tag = [[self.id2label[j] for j in i] for i in labels_pred]

                    true_entities = get_entities(true_tag)
                    pred_entities = get_entities(pred_tag)

                    _, train_entity_score = f1_score(true_entities, pred_entities, labels=self.entity2id)

                    dev_tag_acc, dev_tag_loss, _, dev_entity_score = self._evaluate(self.model, dev_iter, train=True)

                    if dev_best_f1_score < dev_entity_score:
                        dev_best_f1_score = dev_entity_score
                        torch.save(self.model.state_dict(), self.save_model_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},' \
                          'Train Tag Loss: {1:>5.2},  Train Tag Acc: {2:>6.2%}, Train entity F1_Score: {3:>6.2%},' \
                          'Val Tag Loss: {4:>5.2},  Val Tag Acc: {5:>6.2%}, Val entity F1_Score: {6:>6.2%},' \
                          'Time: {7} {8}'
                    self.logger.info(
                        msg.format(total_batch,
                                   loss.item(), train_tag_acc, train_entity_score,
                                   dev_tag_loss, dev_tag_acc, dev_entity_score,
                                   time_dif, improve))

                    self.model.train()
                total_batch += 1
                if total_batch - last_improve > self.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    self.logger.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
            self.scheduler.step()
        time_dif = get_time_dif(start_time)
        self.logger.info("Train time usage:{}".format(time_dif))

    def evaluate_model(self, ):
        start_time = time.time()
        test_iter = self._covert_sentence_label(self.test_path, toy=False)
        self.logger.info("test数据: {}".format(len(test_iter.dataset)))

        test_acc, test_loss, test_report, test_confusion, entity_f1, entity_score = self._evaluate(self.model,
                                                                                                   test_iter, test=True)
        msg = 'Test Tag Loss: {0:>5.2},  Test Tag Acc: {1:>6.2%}, Test Entity F1_Score: {2:>6.2%}'
        self.logger.info(msg.format(test_loss, test_acc, entity_score))
        self.logger.info("Precision, Recall and F1-Score...")
        self.logger.info(test_report)
        # self.logger.info("Confusion Matrix...")
        # self.logger.info(test_confusion)
        self.logger.info("Entity F1_Score...")
        self.logger.info(entity_f1)
        time_dif = get_time_dif(start_time)
        self.logger.info("Test time usage:{}".format(time_dif))

    def _evaluate(self, model, data_iter, train=False, test=False, predict=False):

        loss_total = 0
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for i, (trains, x_lens, labels) in enumerate(data_iter):
                trains = [i.to(self.device) for i in trains]
                labels = labels.to(self.device)
                tag_scores, loss = model(trains, labels=labels)

                pred_label = model.crf_decode(tag_scores, mask=trains[1])
                true_label = [itag[:ilen].tolist() for itag, ilen in zip(labels.cpu().numpy(), x_lens)]

                pred_labels.extend(pred_label)
                true_labels.extend(true_label)
                loss_total += loss

        if predict:
            return pred_labels

        # 标签
        true_flatten = np.hstack([j for i in true_labels for j in i])
        pred_flatten = np.hstack([j for i in pred_labels for j in i])
        tag_acc = metrics.accuracy_score(true_flatten, pred_flatten)

        # 实体
        true_tag = [[self.id2label[j] for j in i] for i in true_labels]
        pred_tag = [[self.id2label[j] for j in i] for i in pred_labels]

        true_entities = get_entities(true_tag)
        pred_entities = get_entities(pred_tag)
        entity_f1, entity_score = f1_score(true_entities, pred_entities, labels=self.entity2id)

        if test:
            report = 0#metrics.classification_report(true_flatten, pred_flatten, target_names=list(self.label2id), digits=4)
            confusion = 0#etrics.confusion_matrix(true_flatten, pred_flatten)
            return tag_acc, loss_total / len(data_iter), report, confusion, entity_f1, entity_score

        if train:
            return tag_acc, loss_total / len(data_iter), entity_f1, entity_score

    def predict_offline(self, ):

        if self.pred_path and os.path.exists(self.pred_path) and self.pred_path.endswith(".csv"):
                df_pred = pd.read_csv(self.pred_path, sep="\t")
                sentences = [Sentence(i) for i in df_pred["sentence"]]
                labels = [Label(dict()) for _ in df_pred["sentence"]]
        else:
            sys.exit("only support csv")

        ds_data = self.modelDataset(self, sentences, labels)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.cpu_count, collate_fn=ds_data.collate_fn)

        pred_labels = self._evaluate(self.model, data_iter, predict=True)
        pred_tag = [[self.id2label[j] for j in i] for i in pred_labels]
        pred_entity = [get_entities(i) for i in pred_tag]

        entity_value = []
        for sentence, entities in zip(sentences, pred_entity):
            tmp = []
            for e, start, end in entities:
                value = sentence.get_text()[start:end+1]
                tmp.append((e, value, start, end))
            entity_value.append(tmp)

        df_label = pd.DataFrame(data=[str(i) for i in pred_tag], columns=['tag_value'])
        df_value = pd.DataFrame(data=[str(i) for i in entity_value], columns=['entity_value'])
        df_result = pd.concat([df_pred, df_value, df_label], axis=1)
        df_result.to_csv(self.save_pred_path, index=False, float_format="%.4f", sep="\t", header=True)
        self.logger.info("predict success...")

    def predict_online(self, sentences_list: List[dict]):
        """传入参数
        # [{
        #     "unique_id": "1234",
        #     "sentence": '开开心心每一天',
        # }]
        """
        self.logger.info("开始预测----")
        sentences = [Sentence(i["sentence"]) for i in sentences_list]
        labels = [Label(dict()) for _ in sentences_list]
        ds_data = self.modelDataset(self, sentences, labels)
        data_iter = DataLoader(ds_data, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.cpu_count, collate_fn=ds_data.collate_fn)
        pred_labels = self._evaluate(self.model, data_iter, predict=True)

        pred_tag = [[self.id2label[j] for j in i] for i in pred_labels]
        pred_entity = [get_entities(i) for i in pred_tag]

        entity_value = []
        for sentence, entities in zip(sentences, pred_entity):
            tmp = []
            for e, start, end in entities:
                value = sentence.get_text()[start:end + 1]
                tmp.append((e, value, start, end))
            entity_value.append(tmp)

        for i, j in zip(sentences_list, entity_value):
            i.update({"pred_entity": str(j)})
        self.logger.info("预测结果: {}".format(sentences_list))
        return sentences_list
