# coding: UTF-8
import json
import threading
import requests
import time
import random

url = "http://127.0.0.1:8008/tagger/predict_online"

REPEAT = 50
THREAD = 10

data = [
    {
        "unique_id": "123",
        "true_label": json.dumps({"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}, ensure_ascii=False),
        "sentence": '彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户,'
    },
    {
        "unique_id": "124",
        "true_label": json.dumps({"organization": {"曼联": [[23, 24]]}, "name": {"温格": [[0, 1]]}}, ensure_ascii=False),
        "sentence": '温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，'
    }
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
    res = json.loads(res.content)
    print(json.dumps(res, indent=2, ensure_ascii=False))

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
      "pred_entity": "[('name', '温格', 0, 1), ('organization', '曼联', 23, 24)]",
      "sentence": "温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，",
      "true_label": "{\"organization\": {\"曼联\": [[23, 24]]}, \"name\": {\"温格\": [[0, 1]]}}",
      "unique_id": "124"
    },
    {
      "pred_entity": "[('name', '温格', 0, 1), ('organization', '曼联', 23, 24)]",
      "sentence": "温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，",
      "true_label": "{\"organization\": {\"曼联\": [[23, 24]]}, \"name\": {\"温格\": [[0, 1]]}}",
      "unique_id": "124"
    }
  ],
  "detail": "",
  "success": "true"
}
"""
