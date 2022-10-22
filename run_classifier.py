# coding: UTF-8
import argparse
import json
from flask import request, jsonify
from script.utils.utils import seed_everything
import script.classifier

parser = argparse.ArgumentParser(description='Chinese Text Task: classifier')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: nn_text_cnn, tf_text_cnn, nn_bert, nn_cnn_bert, nn_ERNIE ...')

parser.add_argument('--train', action='store_true', help='True for train')

parser.add_argument('--adversarial', type=str, required=False, default='base',
                    help='choose a adversarial: base, fgsm, free, pgd')
parser.add_argument('--evaluate', action='store_true', help='True for test')
parser.add_argument('--offline', action='store_true', help='True for predict_offline')
parser.add_argument('--online', action='store_true', help='True for predict_online')
args = parser.parse_args()

if __name__ == '__main__':

    seed_everything()
    model_name = args.model

    if model_name.startswith("nn"):
        seed_everything(666)

    dataset = 'dataset/多分类任务/lizibo'  # 数据集

    kwargs = {
        "dataset": dataset,
        "model_name": model_name,
        "adversarial": args.adversarial,
    }
    Model = eval("script.classifier.{}".format(model_name))

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


        @app.route('/classification/predict_online', methods=["POST"])
        def classification_predict():
            try:
                request_data = json.loads(request.json, encoding='utf-8')
                data = model.predict_online(request_data)
                res, code = {'data': data, "success": "true", 'detail': ''}, 200
            except Exception as e:
                res, code = {'data': 'error', "success": "false", 'detail': 'error {}'.format(str(e))}, 400
            return jsonify(res), code


        app.run(host="0.0.0.0", port=8008, threaded=False)
