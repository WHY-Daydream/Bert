"""
测试脚本：对训练好的 BERT 分类器在验证集/测试集上进行评估
使用方法：
    python3 test.py              # 默认评估验证集 + 测试集
    python3 test.py dev          # 只评估验证集
    python3 test.py test         # 只评估测试集
"""

import sys
import torch
import warnings
warnings.filterwarnings("ignore")

from config import Config
from h1_dataloader_utils import build_dataloader
from h2_bert_classifier_model import BertClassifier
from model2dev_utils import model2dev


def evaluate(dataset_name, model, data_loader, device):
    """调用 model2dev 评估指定数据集"""
    print(f"\n{'='*70}")
    print(f"  数据集: {dataset_name}")
    print(f"{'='*70}")
    report, f1, acc, prec, recall = model2dev(model, data_loader, device)
    print(report)
    print(f"{'='*70}")
    print(f"  总准确率 (Accuracy):  {acc:.6f}")
    print(f"  微平均F1 (F1-score):  {f1:.6f}")
    print(f"  微精确率 (Precision): {prec:.6f}")
    print(f"  微召回率 (Recall):    {recall:.6f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # 1. 加载配置
    conf = Config()

    # 2. 加载模型
    model = BertClassifier().to(conf.device)
    model.load_state_dict(
        torch.load(conf.model_save_path, map_location=conf.device, weights_only=True)
    )
    model.eval()
    print(f"设备: {conf.device}")
    print(f"模型: {conf.model_save_path}")

    # 3. 加载数据
    train_loader, dev_loader, test_loader = build_dataloader()

    # 4. 根据参数决定评估哪个数据集
    args = sys.argv[1:]  # python3 test.py dev / test / (默认都评估)
    if len(args) == 0:
        evaluate("验证集 dev.json (5000 条)", model, dev_loader, conf.device)
        evaluate("测试集 test.json (10000 条)", model, test_loader, conf.device)
    else:
        for arg in args:
            if arg == "dev":
                evaluate("验证集 dev.json (5000 条)", model, dev_loader, conf.device)
            elif arg == "test":
                evaluate("测试集 test.json (10000 条)", model, test_loader, conf.device)
            else:
                print(f"未知参数: {arg}，可用参数: dev, test")
