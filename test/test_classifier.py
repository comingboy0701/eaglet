# coding: UTF-8
import json
import threading
import requests
import time
import random

url = "http://127.0.0.1:8008/classification/predict_online"

REPEAT = 500
THREAD = 10

data = [
    {
        "unique_id": "123",
        "true_label_id": "3",
        "true_label": "education",
        "sentence": '词汇阅读是关键 08年考研暑期英语复习全指南'
    },
    {
        "unique_id": "124",
        "true_label_id": "3",
        "true_label": "education",
        "sentence": '中国人民公安大学2012年硕士研究生目录及书目'
    },
    {
        "unique_id": "124",
        "true_label_id": "3",
        "true_label": "education",
        "sentence": '中国人民公安大学2012年硕士研究生目录及书目'
    },
    {
        "unique_id": "124",
        "true_label_id": "3",
        "true_label": "education",
        "sentence": '中国人民公安大学2012年硕士研究生目录及书目'
    },
]


def run_predict(thread_count):
    time_list = list()
    start = time.time()
    for i in range(REPEAT):
        t = time.time()
        print("线程{}, 正在进行第{}轮请求".format(thread_count, i + 1))
        req_data = random.choices(data, k=16)
        res = requests.post(url, json=json.dumps(req_data, ensure_ascii=False))
        if res.status_code == 200:
            time_list.append(time.time() - t)
    end = time.time()
    print("=" * 80)
    print(time_list)
    print("线程{}, 完成测试, 成功{}次, 总计用时{}, 平均每轮用时{}, 最少用时{}, 最多用时{}".format(
        thread_count, len(time_list), end - start, sum(time_list) / len(time_list), min(time_list), max(time_list)))


if __name__ == "__main__":
    req_data = random.choices(data, k=2)
    res = requests.post(url, json=json.dumps(req_data, ensure_ascii=False))
    start = time.time()
    threads = []
    for i in range(THREAD):
        t = threading.Thread(target=run_predict, args=(i,))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("测试结束, 总计用时{}".format(time.time() - start))

"""
{
  "data": [
    {
      "pred_label": "education",
      "pred_label_id": "3",
      "pred_score": "0.9985",
      "sentence": "词汇阅读是关键 08年考研暑期英语复习全指南",
      "true_label": "education",
      "true_label_id": "3",
      "unique_id": "123"
    },
    {
      "pred_label": "education",
      "pred_label_id": "3",
      "pred_score": "0.9985",
      "sentence": "中国人民公安大学2012年硕士研究生目录及书目",
      "true_label": "education",
      "true_label_id": "3",
      "unique_id": "124"
    }
  ],
  "detail": "",
  "success": "true"
}
"""
