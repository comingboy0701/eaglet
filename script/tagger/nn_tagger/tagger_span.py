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
from typing import List
from tqdm import tqdm

from .tagger import Tagger
from ..metric import get_entities_span, f1_score
import sys


class TaggerSpan(Tagger):

    def train_model(self, ) -> None:
        if self.dev_path and os.path.exists(self.dev_path):
            train_iter = self._covert_sentence_label(self.train_path, self.toy, self.random)
            dev_iter = self._covert_sentence_label(self.dev_path, self.toy, False)
        else:
            train_iter, dev_iter = self._train_test_split(self.train_path)

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
                logits_scores, loss = self.atModel.train(trains, labels, self.optimizer)
                self.optimizer.zero_grad()

                if total_batch % 100 == 0 and total_batch != 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    self.model.eval()

                    # 实体
                    logits_scores = logits_scores.data.cpu().numpy()
                    labels = labels.data.cpu().numpy()

                    logits_scores = [i for i in logits_scores]
                    labels = [i for i in labels]
                    true_entities = get_entities_span(logits_scores, x_lens, self.id2entity)
                    pred_entities = get_entities_span(labels, x_lens, self.id2entity)

                    _, train_entity_score = f1_score(true_entities, pred_entities, self.entity2id)
                    dev_loss, _, dev_entity_score = self._evaluate(self.model, dev_iter, train=True)

                    if dev_best_f1_score < dev_entity_score:
                        dev_best_f1_score = dev_entity_score
                        torch.save(self.model.state_dict(), self.save_model_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter:{0:>6}, ' \
                          'Train Loss:{1:>5.2},  Train entity F1_Score:{2:>6.2%},' \
                          'val Loss:{3:>5.2},  Val entity F1_Score:{4:>6.2%}, ' \
                          'Time:{5} {6}'
                    self.logger.info(
                        msg.format(total_batch,
                                   loss.item(), train_entity_score,
                                   dev_loss, dev_entity_score,
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
        test_loss, test_report, test_confusion, entity_f1, entity_score = self._evaluate(self.model, test_iter,
                                                                                         test=True)
        msg = 'Test Loss: {0:>5.2},Test Entity F1_Score: {1:>6.2%}'
        self.logger.info(msg.format(test_loss, entity_score))
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
        token_lens = []
        with torch.no_grad():
            for i, (trains, x_lens, labels) in enumerate(data_iter):
                trains = [i.to(self.device) for i in trains]
                labels = labels.to(self.device)

                logits_scores, loss = model(trains, labels=labels)

                # 实体
                logits_scores = logits_scores.data.cpu().numpy()
                labels = labels.data.cpu().numpy()

                pred_entities = [get_entities_span(i, j, self.id2entity) for i, j in zip(logits_scores, x_lens)]
                true_entities = [get_entities_span(i, j, self.id2entity) for i, j in zip(labels, x_lens)]

                true_labels.extend(true_entities)
                pred_labels.extend(pred_entities)
                token_lens.extend(x_lens)
                loss_total += loss

        if predict:
            return pred_labels, token_lens

        # 实体 展平计算f1
        token_cumsum = np.cumsum(token_lens)
        pred_entities_flatten = []
        true_entities_flatten = []
        for pred, true, token_len in zip(pred_labels, true_labels, token_cumsum):
            for l, s, e in pred:
                pred_entities_flatten.append((l, s + token_len, e + token_len))
            for l, s, e in true:
                true_entities_flatten.append((l, s + token_len, e + token_len))

        if 0.01 < len(pred_entities_flatten) / len(true_entities_flatten) < 100:
            entity_f1, entity_score = f1_score(true_entities_flatten, pred_entities_flatten, self.entity2id)
        else:
            entity_f1, entity_score = -0.1, -0.1  # 计算量太大，不予计算

        if test:
            report = None  # metrics.classification_report(true_flatten, pred_flatten, target_names=list(entity2id), digits=4)
            confusion = None  # metrics.confusion_matrix(true_flatten, pred_flatten)

            return loss_total / len(data_iter), report, confusion, entity_f1, entity_score

        if train:
            return loss_total / len(data_iter), entity_f1, entity_score

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

        pred_labels, token_lens = self._evaluate(self.model, data_iter, predict=True)
        pred_entity = [get_entities_span(i, token_lens, self.id2entity) for i in pred_labels]

        entity_value = []
        for sentence, entities in zip(sentences, pred_entity):
            tmp = []
            for e, start, end in entities:
                value = sentence.get_text()[start:end + 1]
                tmp.append((e, value, start, end))
            entity_value.append(tmp)

        df_value = pd.DataFrame(data=[str(i) for i in entity_value], columns=['entity_value'])
        df_result = pd.concat([df_pred, df_value], axis=1)
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

        pred_labels, token_lens = self._evaluate(self.model, data_iter, predict=True)
        pred_entity = [get_entities_span(i, token_lens, self.id2entity) for i in pred_labels]

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
