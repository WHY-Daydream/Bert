"""
验证训练好的 BERT 分类器模型
"""
import torch
import warnings
warnings.filterwarnings("ignore")

from config import Config
from h2_bert_classifier_model import BertClassifier

conf = Config()

# 测试样本（覆盖不同意图类别）
test_samples = [
    "这个手机多少钱",
    "我要退货",
    "我的快递到哪了",
    "能开发票吗",
    "推荐一款笔记本电脑",
    "有优惠券吗",
    "我的账号登不上去了",
    "这个商品质量好差",
]

# 加载模型
model = BertClassifier().to(conf.device)
model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device, weights_only=True))
model.eval()

print(f"{'='*60}")
print(f"模型: {conf.model_name}")
print(f"设备: {conf.device}")
print(f"模型权重: {conf.model_save_path}")
print(f"{'='*60}")

# 逐个预测
for text in test_samples:
    output = conf.tokenizer(
        text,
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = output['input_ids'].to(conf.device)
    attention_mask = output['attention_mask'].to(conf.device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    pred_label = conf.class_list[pred_idx]
    print(f"  text: {text}")
    print(f"  预测: {pred_label} (置信度: {confidence:.4f})")
    # 显示 top-3 候选
    top3 = torch.topk(probs[0], 3)
    top3_labels = [(conf.class_list[idx], score.item()) for idx, score in zip(top3.indices, top3.values)]
    top3_str = ", ".join([f"{lb}({sc:.2%})" for lb, sc in top3_labels])
    print(f"  Top-3: {top3_str}")
    print(f"{'-'*60}")
