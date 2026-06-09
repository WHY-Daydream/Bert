"""
Flask 推理服务 - BERT 意图分类
用法:
    cd flask && python3 app.py
    然后浏览器访问 http://localhost:5000
"""

import sys
import os

# 将项目根目录加入路径，使 flask 能 import 上级目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
from config import Config
from h2_bert_classifier_model import BertClassifier

# ========== 加载模型（全局单例，避免每次请求都加载） ==========
conf = Config()
model = BertClassifier().to(conf.device)
model.load_state_dict(
    torch.load(conf.model_save_path, map_location=conf.device, weights_only=True)
)
model.eval()
print(f"[启动] 设备: {conf.device}  |  模型: {conf.model_save_path}")
print(f"[启动] 类别数: {conf.num_classes}")


# ========== Flask 应用 ==========
app = Flask(__name__)


def predict(text):
    """对输入文本进行意图分类，返回预测结果与 top-3 候选"""
    output = conf.tokenizer(
        text,
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = output['input_ids'].to(conf.device)
    attention_mask = output['attention_mask'].to(conf.device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)

    # Top-3
    top3 = torch.topk(probs[0], 3)

    result = {
        'text': text,
        'prediction': conf.class_list[top3.indices[0].item()],
        'confidence': round(probs[0][top3.indices[0]].item(), 6),
        'top3': [
            {
                'label': conf.class_list[idx.item()],
                'score': round(score.item(), 6),
            }
            for idx, score in zip(top3.indices, top3.values)
        ],
    }
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '请提供 text 字段'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'error': '文本不能为空'}), 400
    result = predict(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
