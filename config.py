from setting import ROOT_PATH
import os

tf_bert_config = {
    'bert-base': {
        'config_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_L-12_H-768_A-12/bert_config.json'),
        'checkpoint_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_L-12_H-768_A-12/bert_model.ckpt'),
        'dict_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_L-12_H-768_A-12/vocab.txt'),
        'model_mode': 'bert',
    },
    'roberta-base': {
        'config_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'),
        'checkpoint_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'),
        'dict_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'),
        'model_mode': 'bert',
    },
    'albert-tiny': {
        'config_path': os.path.join(ROOT_PATH, 'bert_model/TF/albert_tiny_google_zh_489k/albert_config.json'),
        'checkpoint_path': os.path.join(ROOT_PATH, 'bert_model/TF/albert_tiny_google_zh_489k/albert_model.ckpt'),
        'dict_path': os.path.join(ROOT_PATH, 'bert_model/TF/albert_tiny_google_zh_489k/vocab.txt'),
        'model_mode': 'albert',
    },
    'simbert': {
        'config_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_simbert_L-12_H-768_A-12/bert_config.json'),
        'checkpoint_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'),
        'dict_path': os.path.join(ROOT_PATH, 'bert_model/TF/chinese_simbert_L-12_H-768_A-12/vocab.txt'),
        'model_mode': 'bert',
    },
}
