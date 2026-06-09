# 该.py文件用于搭建 BiLSTM分类模型的 -> 充当: 学生模型类.

# 导包
import torch                        # 深度学习框架
import torch.nn as nn               # 神经网络模块
from transformers import BertModel, BertTokenizer  # Bert模型, 分词器
from config import Config           # 配置文件类

# todo 1.加载配置文件信息.
conf = Config()     # 后续可以通过 conf. 的形式, 获取配置信息.


# todo 2. 定义BiLSTM分类模型框架
class BiLSTMClassifier(nn.Module):
    # todo 2.1 初始化模型.
    def __init__(self):
        # 1. 继承父类初始化方法
        super().__init__()

        # 2. 定义嵌入层, 将词的整数id转换成(稠密)向量表示.
        # 输入维度: config.bert_config.vocab_size(21128, 词汇表大小)
        # 输出维度: config.embed_size
        self.embedding = nn.Embedding(conf.bert_config.vocab_size, conf.embed_size)

        # 3. 定义双向LSTM层(BiLSTM)
        # conf.embed_size: 输入特征维度(即: 词向量维度)
        # hidden_size_lstm: LSTM隐藏层维度
        # num_layers: LSTM层数
        # batch_first: 输入张量的维度顺序为: [batch_size, seq_len, input_size]
        # bidirectional: 是否启用双向LSTM(前后两个方向都计算, 输出维度会翻倍)
        self.lstm = nn.LSTM(conf.embed_size, conf.hidden_size_lstm, conf.num_layers, batch_first=True, bidirectional=True)

        # 4. 随机失活层.
        self.dropout = nn.Dropout(conf.dropout)

        # 5. 定义全连接分类层, 输入维度:双向LSTM输出维度翻倍, 输出维度: conf.num_classes(10个类别)
        self.fc = nn.Linear(conf.hidden_size_lstm * 2, conf.num_classes)

    # todo 2.2 定义前向传播方法.
    def forward(self, input_ids, attention_mask):
        # 嵌入层
        embed = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]

        # 使用 attention_mask 掩码填充 token 的嵌入（核心处理）
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        embed = embed * attention_mask  # 将填充 token 的嵌入置为 0

        # LSTM 层
        lstm_out, (hidden, _) = self.lstm(embed)  # lstm_out: [batch_size, seq_len, hidden_size*2]

        # 取最后一个有效 token（非 padding）位置的隐藏状态
        seq_lengths = attention_mask.squeeze(-1).sum(dim=1).long()  # [batch_size]
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        hidden = lstm_out[batch_indices, seq_lengths - 1, :]  # [batch_size, hidden_size*2]

        # Dropout 和全连接层
        hidden = self.dropout(hidden)  # [batch_size, hidden_size*2]
        logits = self.fc(hidden)  # [batch_size, num_classes]

        #  13. 返回分类结果.
        return logits



# todo 3.测试代码
if __name__ == '__main__':
    # 1. 创建BiLSTM模型师实例.
    model = BiLSTMClassifier()
    # 2. 准备示例文本, 用于测试 模型的输入数据.
    texts = ['王者荣耀', '今天天气很好']
    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)

    # 3. 编码文本 -> 将原始文本转成模型所需要的 的输入数据(Token ID, Attention Mask)
    encode_inputs = tokenizer(
        texts,                      # 待编码的文本列表
        padding='max_length',       # 填充至最大长度.
        max_length = 100,           # 最大序列长度
        return_tensors='pt'         # 返回PyTorch张量
    )

    # 4. 提取模型输入张量: 从编码结果中拿出 Token ID 和 Attention Mask张量.
    input_ids = encode_inputs['input_ids']
    attention_mask = encode_inputs['attention_mask']
    print(f'input_ids: {input_ids}')
    print(f'attention_mask: {attention_mask}')
    print('-' * 40)

    # 5. 创建自定义的BERT分类模型
    logits = model(input_ids, attention_mask)
    print(f'logits: {logits}')      # 未归一化的分类得分(每行对应1个样本, 每列对应1个类别)
    print('-' * 40)

    # 7. 计算类别概率, 对logits做softmax()归一化, 得到每个类别在[0, 1]区间的概率
    probs = torch.softmax(logits, dim=-1)
    print(f'logits: {logits}')
    print('-' * 40)

    # 8. 获取预测分类: 即概率最大的类别索引.
    preds = torch.argmax(logits, dim=-1)
    print(f'logits: {preds}')        # 最终结果: 每个样本的预测类别索引.
