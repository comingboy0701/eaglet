# eaglet 常用算法比较

## 1 分类模型
 
### 1.1 使用说明 

`python3 run_classifier.py --model nn_bert --adversarial fgm --train --evaluate`

`python3 run_classifier.py --model nn_textcnn --adversarial  pgd --train --evaluate`

- 模型类型： textcnn, bert,nezha, bertcnn, bertrnn...
- 训练扰动：fgm, pgd, fgsm, free, freelb

### 1.2 实验对比

- 机器配置: 显卡v100, 内存128G, cpu32核

- 数据

| dataset | class | train | dev | test |备注|
| :--------:  | :--------:  | :--------:  | :--------:  |:--------:  |:--------: |
| THUCNews  | 10类(每类2万条) | 18万  | 1万 | 1万 | 文本长度在20到30之间 |


|model+method|acc|micro-precison|micro-recall|micro-f1|训练时间|epoch(20)|Test loss|实验配置|
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|textcnn|89.18%|0.8924|0.8918|0.8919|1分23秒|3|0.34|early stop|
|textcnn+fgm|89.18%|0.8924|0.8918|0.8919|1分23秒|3|0.34|early stop|
|textcnn+fgsm|90.87%|0.9089|0.9087|0.9086|5分11秒|6|0.3|epsilon=0.1,early stop|
|textcnn+pgd|89.81%|0.8989|0.8981|0.8982|5分12秒|4|0.33|epsilon=0.1,K=3,alpha=0.1,early stop|
|textcnn+free|88.07%|0.8817|0.8807|0.8808 |3分51秒|3|0.39|epsilon=0.1,M=3,early stop|
| bert_base  | 94.31%|0.9431|0.9431|0.9429|16分59秒|3|0.17|early stop|
| bert_base+fgm  | 94.74%|0.9476|0.9474|0.9473|30分19秒|3|0.17|early stop|
| bert_base+pgd  | 94.64%|0.9467|0.9464|0.9464|52分47秒|3|0.16|early stop|
| nezha  | 94.11%|0.9416|0.9411|0.9410|13分54秒|2|0.18|early stop|
| nezha+fgm  | 95.09%|0.9509|0.9509|0.9508|35分59秒|3|0.15|early stop|
| nezha+pgd  |  | 
| robert  |  | 
| robert+fgm  |  | 
| robert+pgd  |  | 
 
 - 备注：训练时间可信度不高，由于跑model的时候可能有其他模型在run

## 2 实体识别模型

| dataset | class | train | dev | test |备注|
| :--------:  | :--------:  | :--------:  | :--------:  |:--------:  |:--------: |
| CLUENER细粒度命名实体识别  | 10个标签类别 | 9673条  | 1075条 | 1343条 |  |

### 2.1 使用说明 

`python3 run_tagger.py  --model nn_biLstm_crf --adversarial base --train --evaluate`

`python3 run_tagger.py  --model nn_bert_crf --adversarial  fgm --train --evaluate`

`python3 run_tagger.py  --model nn_bert_gp --adversarial  fgm --train --evaluate`

 - 备注：训练时间可信度不高，由于跑model的时候可能有其他模型在run
 
### 2.2 实验对比

|model+method|Entity-F1_Score|训练时间|epoch(20)|Test loss|实验配置|
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|biLstm_crf|  66.09% | 4分22秒  | 14 |   4.7e+02 |  early stop | 
|biLstm_crf+fgm|  65.59% | 10分14秒  | 17 |  4.7e+02 |  epsilon=0.1 early stop | 
|biLstm_crf+fgsm|  71.98% | 10分27秒  | 17 |  2.5e+02 |  epsilon=0.1 early stop | 
|bert_base_crf|  78.86% | 12分42秒  | 17 |  8.7e+01 |  early stop | 
|bert_base_crf+fgm |  78.39% | 27分58秒  | 20 |   1e+02 |  epsilon=0.1 early stop | 
|bert_base_crf+fgsm |  77.98% | 36分54秒  | 27 |  1.1e+02 |  epsilon=0.1 early stop | 
|nezha_crf |  79.39% | 22分44秒  | 17 |  9.1e+01 |  early stop | 
|nezha_crf+fgsm |  79.75%| 52分45秒  | 20 |  9.9e+01 | epsilon=0.1 early stop |
|nezha_gp|  79.33% | 42分50秒  | 30 |  0.42 |  early stop | 
|nezha_gp+fgsm|  79.99% | 73分50秒  | 30 |  0.35 | epsilon=0.1 early stop |
|nezha_gp+fgm|  79.56% | 71分07秒  | 30 |  0.44 | epsilon=0.1 early stop |  