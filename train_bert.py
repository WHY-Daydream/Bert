"""
BERT 分类器训练脚本（教师模型）
- 加载本地缓存的 chinese-macbert-base
- 在意图分类数据上微调
- 保存最佳模型供后续蒸馏使用
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

from config import Config
from h1_dataloader_utils import build_dataloader
from h2_bert_classifier_model import BertClassifier
from model2dev_utils import model2dev


def train_bert():
    # 1. 加载配置
    conf = Config()
    print(f"设备: {conf.device}")
    print(f"类别数量: {conf.num_classes}")
    print(f"类别列表: {conf.class_list}")
    print(f"模型路径: {conf.bert_path}")

    # 2. 准备数据加载器
    train_loader, dev_loader, test_loader = build_dataloader()
    print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(dev_loader)}, 测试集批次: {len(test_loader)}")

    # 3. 创建模型
    model = BertClassifier().to(conf.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # 5. 创建保存目录
    save_dir = os.path.dirname(conf.model_save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 6. 初始化最佳指标
    best_f1 = 0.0

    # 7. 训练循环
    for epoch in range(conf.num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.num_epochs}")):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # 每500步评估一次
            if i % 500 == 0 and i > 0:
                avg_loss = total_loss / batch_count
                report, f1score, accuracy, precision, recall = model2dev(model, dev_loader, conf.device)
                print(f"\nEpoch {epoch+1}, Step {i}, Loss: {avg_loss:.4f}")
                print(f"验证集 - F1: {f1score:.4f}, 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")

                # 保存最佳模型
                if f1score > best_f1:
                    torch.save(model.state_dict(), conf.model_save_path)
                    best_f1 = f1score
                    print(f"✅ 保存最佳模型, F1={f1score:.4f}")

                model.train()

        # 每个epoch结束评估
        avg_loss = total_loss / batch_count
        report, f1score, accuracy, precision, recall = model2dev(model, dev_loader, conf.device)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        print(f"验证集 - F1: {f1score:.4f}, 准确率: {accuracy:.4f}")
        print(f"验证集评估报告:\n{report}")
        print(f"{'='*60}")

        if f1score > best_f1:
            torch.save(model.state_dict(), conf.model_save_path)
            best_f1 = f1score
            print(f"✅ 保存最佳模型, F1={f1score:.4f}")

    # 8. 加载最佳模型在测试集上评估
    print(f"\n{'='*60}")
    print("在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    report, f1score, accuracy, precision, recall = model2dev(model, test_loader, conf.device)
    print(f"测试集 - F1: {f1score:.4f}, 准确率: {accuracy:.4f}")
    print(f"测试集评估报告:\n{report}")
    print(f"模型已保存到: {conf.model_save_path}")


if __name__ == '__main__':
    train_bert()
