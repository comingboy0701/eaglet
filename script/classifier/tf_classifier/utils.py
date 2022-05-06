import keras
from keras.preprocessing.sequence import pad_sequences
from .snippets import DataGenerator, sequence_padding


class TextCnnDataset(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size=64, max_len=128, *args, **kwargs):
        super().__init__(data, batch_size)
        self.max_len = max_len
        self.word2index_dict = kwargs['word2index_dict']

    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        for is_end, (text, label) in self.sample(random):
            text, label = text.get_text(), label.get_label()
            token_ids = [self.word2index_dict[i] if i in self.word2index_dict else 0 for i in text]
            label = int(label)  # self.label2index_dict.get(label, 0)  # 不存在就返回一个默认值，为了预测时构造数据迭代器
            batch_token_ids.append(token_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids, self.max_len)
                batch_labels = pad_sequences(batch_labels)
                yield batch_token_ids, batch_labels
                batch_token_ids, batch_labels = [], []


class BertDataset(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size=64, max_len=128, *args, **kwargs):
        super().__init__(data, batch_size)
        self.max_len = max_len
        self.tokenizer = kwargs['tokenizer']

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text = text.get_text()
            label = int(label.get_label())
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_generator, dev_best_loss, logger, save_model_path, patience):
        super().__init__()
        self.logger = logger
        self.last_improve = 0
        self.save_model_path = save_model_path
        self.dev_best_loss = dev_best_loss
        self.patience = patience
        self.valid_generator = valid_generator

    def on_epoch_end(self, epoch, logs=None):
        valid_loss, valid_acc = self.model.evaluate_generator(self.valid_generator.forfit(),
                                                              steps=len(self.valid_generator))
        if valid_loss < self.dev_best_loss:
            self.dev_best_loss = valid_loss
            self.model.save_weights(self.save_model_path)
            improve = '*'
            self.last_improve = epoch
        else:
            improve = ''
        self.logger.info('epoch:{0}, dev_best_loss:{1:>5.2}, valid_acc:{2:>6.2%}, {3}' \
                         .format(epoch, self.dev_best_loss, valid_acc, improve))
        if epoch - self.last_improve >= self.patience:
            self.model.stop_training = True

