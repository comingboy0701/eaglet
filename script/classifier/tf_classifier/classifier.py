import numpy as np
import pandas as pd
import time
from sklearn import metrics
from script.utils.utils import get_time_dif
from script import Sentence, Label
from typing import List
from sklearn.model_selection import train_test_split
from .utils import Evaluator


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
        self.loss_func = None
        self.optimizer = None
        self.max_len = None
        self.word2index = None
        self.metric_func = None
        self.patience = None
        self.tokenizer = None

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
        ds_data = [[i, j] for i, j in zip(sentences, labels)]
        kwargs = {
            "tokenizer": self.tokenizer,
            "word2index_dict": self.word2index,
        }
        train_iter = self.modelDataset(data=ds_data,
                                       batch_size=self.batch_size,
                                       max_len=self.max_len,
                                       **kwargs
                                       )
        return train_iter

    def train_model(self, ) -> None:

        self.model.summary(print_fn=self.logger.info)

        train_iter = self._covert_sentence_label(self.train_path, self.toy)
        dev_iter = self._covert_sentence_label(self.dev_path, self.toy)

        start_time = time.time()
        dev_best_loss = float('inf')

        self.model.fit_generator(
            generator=train_iter.forfit(),
            steps_per_epoch=len(train_iter),
            epochs=self.num_epochs,
            callbacks=[Evaluator(dev_iter, dev_best_loss, self.logger, self.save_model_path, self.patience)]
        )
        time_dif = get_time_dif(start_time)
        self.logger.info("Train time usage:{}".format(time_dif))
        self.model.load_weights(self.save_model_path)

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

    def _evaluate(self, model, data_iter, test=False, predict=False):

        pred_list = [model.predict(x) for x, y in data_iter]
        pred = np.concatenate(pred_list, axis=0)

        pred_labels = np.argsort(pred, axis=1)[:, ::-1][:, :self.topk]
        pred_scores = np.sort(pred, axis=1)[:, ::-1][:, :self.topk]

        if predict:
            return np.concatenate(pred_labels).astype("int8"), np.concatenate(pred_scores).astype("float16")

        true_labels = np.concatenate([y for _, y in data_iter])
        # loss = self.loss_func(true_labels, pred)
        # acc = self.metric_func(true_labels, pred)
        loss, acc = model.evaluate_generator(data_iter.forfit(), steps=len(data_iter))
        pred_labels = pred_labels[:, 0]
        if test:
            report = metrics.classification_report(true_labels, pred_labels, target_names=list(self.label2id),
                                                   digits=4)
            confusion = metrics.confusion_matrix(true_labels, pred_labels)
            return acc, loss, report, confusion

    def predict_offline(self, ):

        df_pred = pd.read_csv(self.pred_path)
        sentences = [Sentence(i) for i in df_pred["sentence"]]
        labels = [Label("0") for _ in df_pred["sentence"]]
        ds_data = [[i, j] for i, j in zip(sentences, labels)]
        data_iter = self.modelDataset(data=ds_data,
                                      batch_size=self.batch_size,
                                      max_len=self.max_len,
                                      word2index_dict=self.word2index,
                                      label2index_dict=self.label2id)

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
        #     "sentence": '开开心心每一天',
        # }]
        """
        self.logger.info("预测数据: {}".format(sentences_list))
        sentences = [Sentence(i["sentence"]) for i in sentences_list]
        labels = [Label("0") for _ in sentences_list]
        ds_data = [[i, j] for i, j in zip(sentences, labels)]
        data_iter = self.modelDataset(data=ds_data,
                                      batch_size=self.batch_size,
                                      max_len=self.max_len,
                                      word2index_dict=self.word2index,
                                      label2index_dict=self.label2id)
        pred_labels, pred_scores = self._evaluate(self.model, data_iter, predict=True)
        for i, j, k in zip(sentences_list, pred_labels, pred_scores):
            i.update({"pred_label_id": str(j[0]), 'pred_label': self.id2label[j[0]], "pred_score": str(round(k[0], 4))})
        return sentences_list
