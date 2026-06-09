# BERT 意图分类系统

基于 **Chinese-MacBERT-Base** 微调的电商客服意图分类引擎，支持 20 种意图识别，并提供 Flask Web 服务。

## 效果

| 数据集 | 样本数 | 准确率 | F1 分数 |
|--------|:------:|:------:|:-------:|
| 验证集 (dev.json) | 5,000 | **99.90%** | **0.9990** |
| 测试集 (test.json) | 10,000 | **99.89%** | **0.9989** |

## 支持的 20 种意图

| # | 意图 | # | 意图 | # | 意图 | # | 意图 |
|:-:|:----|:-:|:----|:-:|:----|:-:|:----|
| 0 | 价格查询 | 5 | 售后咨询 | 10 | 安装咨询 | 15 | 订单查询 |
| 1 | 优惠活动查询 | 6 | 商品咨询 | 11 | 库存查询 | 16 | 评价咨询 |
| 2 | 会员咨询 | 7 | 商品推荐 | 12 | 投诉建议 | 17 | 账号咨询 |
| 3 | 发票咨询 | 8 | 商品搜索 | 13 | 支付咨询 | 18 | 退换货申请 |
| 4 | 发货查询 | 9 | 商品比较 | 14 | 物流查询 | 19 | 退款申请 |

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（推荐，CPU 也可运行但较慢）

### 安装依赖

```bash
pip install torch transformers tqdm scikit-learn flask
```

### 下载预训练模型

模型会自动从 HuggingFace 下载并缓存，也可以使用本地缓存（如 ModelScope）：

```bash
# 方式一：自动下载（首次运行会自动拉取）
python3 -c "from config import Config; Config()"

# 方式二：使用本地缓存（修改 config.py 中 bert_path）
# self.bert_path = "./.cache/modelscope/models/hfl/chinese-macbert-base"
```

### 训练 BERT 分类器（教师模型）

```bash
python3 train_bert.py
```

训练好的模型保存在 `save_models/bert_classifier_model.pt`。

### 评估模型

```bash
# 评估验证集 + 测试集
python3 test.py

# 只评估验证集
python3 test.py dev

# 只评估测试集
python3 test.py test
```

## 知识蒸馏

训练 BiLSTM 学生模型，用于轻量部署：

### 硬标签蒸馏

```bash
python3 h4_hard_label_distillation.py
```

### 软标签蒸馏

```bash
python3 h5_soft_label_distillation.py
```

### BiLSTM 预测

```bash
python3 -c "
from h6_bilstm_predict_fun import predict_fun
print(predict_fun({'text': '这个手机多少钱'}))
"
```

## Web 服务（Flask）

启动服务：

```bash
python3 flask_app/app.py
```

浏览器访问 `http://localhost:5000`

### API 调用

```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"我要退货"}' | python3 -m json.tool
```

返回：

```json
{
    "prediction": "退换货申请",
    "confidence": 0.9922,
    "text": "我要退货",
    "top3": [
        {"label": "退换货申请", "score": 0.9922},
        {"label": "退款申请", "score": 0.0009},
        {"label": "支付咨询", "score": 0.0009}
    ]
}
```

## 项目结构

```
├── config.py                         # 配置文件（模型路径、训练参数）
├── train_bert.py                     # BERT 分类器训练脚本
├── test.py                           # 模型评估脚本
├── model2dev_utils.py                # 评估函数（分类报告 + 指标）
├── verify_model.py                   # 单条预测验证脚本
│
├── h1_dataloader_utils.py            # 数据加载 & DataLoader 构建
├── h2_bert_classifier_model.py       # BERT 分类模型（教师模型）
├── h3_bilstm_classifier_model.py     # BiLSTM 分类模型（学生模型）
├── h4_hard_label_distillation.py     # 硬标签知识蒸馏
├── h5_soft_label_distillation.py     # 软标签知识蒸馏
├── h6_bilstm_predict_fun.py          # BiLSTM 预测函数
│
├── flask_app/
│   ├── app.py                        # Flask 推理服务
│   └── templates/
│       └── index.html                # 前端页面
│
├── data/
│   ├── train.json                    # 训练集（10,000 条）
│   ├── dev.json                      # 验证集（5,000 条）
│   └── test.json                     # 测试集（10,000 条）
│
└── intents_temp.txt                  # 意图类别列表
```

## 数据集格式

每行一个 JSON 对象：

```json
{"text": "这个手机多少钱", "intent": "价格查询", "difficulty": "easy", "source": "ecommerce-train"}
{"text": "我要退货", "intent": "退款申请", "difficulty": "hard", "source": "ecommerce-train"}
```

## 技术栈

- **框架**: PyTorch + Transformers
- **预训练模型**: [hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)
- **Web 服务**: Flask
- **评估指标**: Accuracy, Precision, Recall, F1-score
