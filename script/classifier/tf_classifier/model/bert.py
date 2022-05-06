from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.layers import Dense, Lambda
import os
from script.adversarial.fgm_model_tf import adversarial_training
from script.classifier.tf_classifier.classifier import Classifier
from script.classifier.setting import DataConfig
from script.classifier.tf_classifier.utils import BertDataset
from config import tf_bert_config

set_gelu('tanh')  # 切换gelu版本


class Config(DataConfig):
    """数据配置参数"""

    def __init__(self, **kwargs):
        DataConfig.__init__(self,
                            dataset=kwargs.get('dataset'),
                            cache_dir=kwargs.get('cache_dir'),
                            model_name=kwargs.get('model_name'),
                            adversarial=kwargs.get('adversarial'),
                            )
        self.patience = 3  # 若超过3 epoch效果还没提升，则提前结束训练
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_config = tf_bert_config["bert-base"]

        self.dict_path = self.bert_config["dict_path"]
        self.checkpoint_path = self.bert_config["checkpoint_path"]
        self.config_path = self.bert_config["config_path"]
        self.model_mode = self.bert_config["model_mode"]


class BertClassifier(Classifier, Config):
    def __init__(self, **kwargs):
        Classifier.__init__(self, **kwargs)
        Config.__init__(self, **kwargs)
        self._set_optimizer()
        self._set_loss_func()
        self._set_metric_func()
        self._set_optimizer()
        self.build_model()
        self.modelDataset = BertDataset

        adversarial = kwargs.get('adversarial', 'base')
        if adversarial == 'fgm':
            # 写好函数后，启用对抗训练只需要一行代码
            adversarial_training(self.model, 'Embedding-Token', 0.5)
            self.logger.info("启用 adversarial: {}".format(adversarial))

        if kwargs.get('evaluate') and os.path.exists(self.save_model_path):
            self.model.load_weights(self.save_model_path)
            self.logger.info("load {} success".format(self.model_name))

    def build_model(self, ) -> None:
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            return_keras_model=False,
            model=self.model_mode)
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.output)
        output = Dense(
            units=self.num_classes,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)
        self.model = keras.models.Model(bert.input, output)
        self.model.compile(
            loss=self.loss_func,  # 'sparse_categorical_crossentropy',
            optimizer=self.optimizer,  # Adam(self.learning_rate),
            metrics=[self.metric_func]  # ['accuracy'],
        )

    def _set_optimizer(self, ):
        self.optimizer = Adam(self.learning_rate)  # keras.optimizers.adam()

    def _set_loss_func(self):
        self.loss_func = keras.losses.SparseCategoricalCrossentropy()

    def _set_metric_func(self):
        self.metric_func = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
