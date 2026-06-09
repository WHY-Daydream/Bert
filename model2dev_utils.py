import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。
    参数：
        model (nn.Module): BERT 分类模型。
        data_loader (DataLoader): 数据加载器（验证或测试集）。
        device (str): 设备（"cuda" 或 "cpu"）。
    返回：
        tuple: (分类报告, F1 分数, 准确度, 精确度，召回率)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度
            - recall: 微平均召回率
    """


    # todo 1. 设置模型为评估模式（禁用 dropout,并改变batch_norm行为）
    model.eval()
    # 2. 初始化列表，all_preds, all_labels, 存储预测结果和真实标签
    all_preds = [] # 保存预测的标签
    all_labels = [] # 保存真实的标签
    # 3. todo torch.no_grad()禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 4. 遍历数据加载器，逐批次进行预测
        for batch in tqdm(data_loader):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels =batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # 4.2 前向传播：模型预测
            logits = model(input_ids, attention_mask)

            # 4.3 获取预测结果（最大 logits分数 对应的类别）
            y_pred = torch.argmax(logits, dim=-1)

            # 4.4 存储预测和真实标签
            all_preds.extend(y_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 5. 计算分类报告、F1 分数、准确率，精确率，召回率
    report = classification_report(all_labels, all_preds)
    f1score = f1_score(all_labels, all_preds, average='micro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')

    # 6. 返回评估结果
    return report, f1score, accuracy, precision, recall


if __name__ == '__main__':
    # 调用示例：使用 model2dev 评估验证集和测试集
    import warnings
    warnings.filterwarnings("ignore")

    from config import Config
    from h1_dataloader_utils import build_dataloader
    from h2_bert_classifier_model import BertClassifier

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
    _, dev_loader, test_loader = build_dataloader()

    # 4. 评估验证集
    print(f"\n{'='*60}")
    print("验证集 (dev.json) 评估")
    print(f"{'='*60}")
    report, f1, acc, prec, recall = model2dev(model, dev_loader, conf.device)
    print(report)
    print(f"F1: {f1:.6f}  Acc: {acc:.6f}  Prec: {prec:.6f}  Recall: {recall:.6f}")

    # 5. 评估测试集
    print(f"\n{'='*60}")
    print("测试集 (test.json) 评估")
    print(f"{'='*60}")
    report, f1, acc, prec, recall = model2dev(model, test_loader, conf.device)
    print(report)
    print(f"F1: {f1:.6f}  Acc: {acc:.6f}  Prec: {prec:.6f}  Recall: {recall:.6f}")
