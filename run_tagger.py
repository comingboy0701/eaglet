# coding: UTF-8
import argparse
import json
from flask import request, jsonify
from script.utils.utils import seed_everything
import script.tagger


parser = argparse.ArgumentParser(description='Chinese Text Task: tagger')
parser.add_argument('--model', type=str, required=True, help='choose a model: nn_biLstm_crf, nn_bert_crf, nn_bert_gp...')
parser.add_argument('--train', action='store_true', help='True for train')
parser.add_argument('--adversarial', type=str, required=False, default='base', help='choose a adversarial: base, fgsm, free, pgd')
parser.add_argument('--evaluate', action='store_true', help='True for test')
parser.add_argument('--offline', action='store_true', help='True for predict_offline')
parser.add_argument('--online', action='store_true', help='True for predict_online')
args = parser.parse_args()


if __name__ == '__main__':

    seed_everything()
    model_name = args.model

    if model_name.startswith("nn"):
        seed_everything(666)

    dataset = 'dataset/实体提取/cluener_public'  # 数据集
    # dataset = 'dataset/实体提取/京东2022'  # 数据集

    kwargs = {
        "dataset": dataset,
        "model_name": model_name,
        "adversarial": args.adversarial,
    }
    Model = eval("script.tagger.{}".format(model_name))

    if args.train:
        kwargs["evaluate"] = False
        model = Model(**kwargs)
        model.train_model()

    if args.evaluate:
        kwargs["evaluate"] = True
        model = Model(**kwargs)
        model.evaluate_model()

    if args.offline:
        kwargs["evaluate"] = True
        model = Model(**kwargs)
        model.predict_offline()

    if args.online:
        kwargs["evaluate"] = True
        model = Model(**kwargs)

        from flask import Flask
        app = Flask(__name__)

        @app.route("/healthCheck")
        def status():
            return "success", 200

        @app.route('/tagger/predict_online',  methods=["POST"])
        def classification_predict():
            try:
                request_data = json.loads(request.json, encoding='utf-8')
                data = model.predict_online(request_data)
                res, code = {'data': data, "success": "true", 'detail': ''}, 200
            except Exception as e:
                res, code = {'data': 'error', "success": "false", 'detail': 'error {}'.format(str(e))}, 400
            return jsonify(res), code

        app.run(host="0.0.0.0", port=8008, threaded=False)




