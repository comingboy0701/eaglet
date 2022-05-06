# coding: UTF-8
import os

from script.adversarial.fgm_model_tf import adversarial_training
from script.classifier.tf_classifier.classifier import Classifier
from script.classifier.tf_classifier.utils import TextCnnDataset
from script.classifier.setting import DataConfig
from script.pretraining import load_emb_vec
from script.pretraining import Vocabulary
import keras
import numpy as np


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
            self.init_embedding, self.word2index = load_emb_vec(self.emb_path)
        else:
            vocabulary = Vocabulary(train_files=[self.train_path, self.test_path, self.dev_path],
                                    emb_path=self.emb_path,
                                    vocab_path=self.vocab_path,
                                    max_vocab_size=10000)
            self.word2index = vocabulary.get_vocab()
            self.init_embedding = np.random.random((len(self.word2index), 300))

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000 batch效果还没提升，则提前结束训练
        self.patience = 3  # 若超过3 epoch效果还没提升，则提前结束训练
        self.n_vocab = len(self.word2index)+1  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 32  # 每句话处理成的长度(短填长切)
        self.embedding_trainable = True
        self.learning_rate = 1e-3  # 学习率
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256


class TextCNNClassifier(Classifier, Config):
    def __init__(self, **kwargs):
        Classifier.__init__(self, **kwargs)
        Config.__init__(self, **kwargs)
        self._set_optimizer()
        self._set_loss_func()
        self._set_metric_func()
        self._set_optimizer()
        self.build_model()
        self.modelDataset = TextCnnDataset

        adversarial = kwargs.get('adversarial', 'base')

        if adversarial == 'fgm':
            # 写好函数后，启用对抗训练只需要一行代码
            adversarial_training(self.model, 'Embedding-Token', 0.5)
            self.logger.info("启用 adversarial: {}".format(adversarial))

        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_weights(self.save_model_path)
            self.logger.info("load {} success".format(self.model_name))

    def build_model(self):
        inputs = keras.layers.Input(shape=(self.max_len,))
        x = keras.layers.Embedding(self.init_embedding.shape[0],
                                   output_dim=self.init_embedding.shape[1],
                                   weights=[self.init_embedding],
                                   input_length=self.max_len,
                                   trainable=self.embedding_trainable,
                                   name="Embedding-Token"
                                   )(inputs)
        cnn_list = []
        for i in range(len(self.filter_sizes)):
            _cnn = keras.layers.Conv1D(256, self.filter_sizes[i], activation='relu', padding='same')(x)
            _cnn = keras.layers.GlobalAveragePooling1D()(_cnn)
            cnn_list.append(_cnn)
        cnn = keras.layers.concatenate(cnn_list, axis=-1)
        dropout = keras.layers.Dropout(self.dropout)(cnn)
        dense = keras.layers.Dense(self.num_classes, activation='softmax')(dropout)
        self.model = keras.models.Model(inputs=[inputs], outputs=[dense])
        self.model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=[self.metric_func])

    def _set_optimizer(self, ):
        self.optimizer = keras.optimizers.adam()

    def _set_loss_func(self):
        self.loss_func = keras.losses.SparseCategoricalCrossentropy()

    def _set_metric_func(self):
        self.metric_func = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
