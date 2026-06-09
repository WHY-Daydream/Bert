# 该.py文件用于搭建 BERT分类模型的.

# 导包
import torch  # 深度学习框架
import torch.nn as nn  # 神经网络模块
from transformers import BertModel, BertTokenizer  # Bert模型, 分词器
from config import Config  # 配置文件类

# todo 1.加载配置文件信息.
conf = Config()  # 后续可以通过 conf. 的形式, 获取配置信息.


# todo 2. 定义BERT分类模型框架
class BertClassifier(nn.Module):
    # todo 2.1 初始化模型.
    def __init__(self):
        # 1. 继承父类初始化方法
        super().__init__()
        # 2. 加载BERT模型
        self.bert = conf.bert_model
        # 3. 定义全连接分类层, 输入维度: 768(BERT的隐藏层维度), 输出维度: conf.num_classes(10个类别)
        # self.fc = nn.Linear(768, 10)
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)
        pass

    # todo 2.2 定义前向传播方法.
    def forward(self, input_ids, attention_mask):
        # 1. 将Token ID 和 注意力掩码输入BERT模型, 获取模型输出(包含: last_hidden_state, pooler_output)
        # input_ids: 输入的Token ID张量, 形状为: [batch_size, 序列长度max_length]
        # attention_mask: 输入的注意力掩码张量, 形状为: [batch_size, 序列长度max_length]
        hidden_state, pooler_output = self.bert(input_ids, attention_mask, return_dict=False)
        # print(f"hidden_state的形状: {hidden_state.shape}") # torch.Size([2, 30, 768]),
        # print(f"pooler_output的形状: {pooler_output.shape}") # torch.Size([2, 768])
        # print("#######################################################")
        # 2. 取BERT的 pooler_output([CLS] token的隐藏状态,经过一层全连接 + Tanh激活,  即: 样本属于每个分类的概率) 作为句子的整体表示, 输入分类层.
        logits = self.fc(pooler_output)
        # print(f"logits的形状: {logits.shape}") # torch.Size([2, 10])
        # 3. 返回分类结果.
        return logits


# todo 3.测试代码
if __name__ == '__main__':
    # 1. 实例化模型
    model = BertClassifier()
    # 2. 准备示例文本, 用于测试 模型的输入数据.
    texts = ['文本分类', '今天天气很好']

    # 3. 编码文本 -> 将原始文本转成模型所需要的 的输入数据(Token ID, Attention Mask)
    output = conf.tokenizer(
        texts,
        add_special_tokens=True, # 添加[CLS]...
        padding='max_length', # 填充方式
        max_length=conf.pad_size, # 最大长度
        truncation=True, # 截断
        return_attention_mask=True, # 返回注意力掩码
        return_tensors='pt' # 输入张量格式
    )
    # 4. 提取模型输入张量: 从编码结果中拿出 Token ID 和 Attention Mask张量.
    # 5. 模型前向传播, 获取模型输出.
    logits = model(output['input_ids'], output['attention_mask'])
    # 7. 计算类别概率, 对logits做softmax()归一化, 得到每个类别在[0, 1]区间的概率
    logits_prob = torch.softmax(logits, dim=-1)
    # 8. 获取预测分类: 即概率最大的类别索引.
    pred_label = torch.argmax(logits_prob, dim=-1)
    for label in pred_label.tolist():
        print({k:v for k, v in enumerate(conf.class_list)}[label])