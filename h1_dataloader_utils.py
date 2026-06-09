# 该.py文件的作用是 -> 获取到 数据集加载器(DataLoader)

# 导包
from tqdm import tqdm  # 进度条
import torch  # 深度学习框架
from torch.utils.data import Dataset, DataLoader  # 数据集对象, 数据加载器对象.
from transformers import BertTokenizer  # BERT分词器
import time  # 时间处理
from config import Config  # 配置文件类
import json  # 用于读取 JSON Lines 数据

# 创建配置文件对象
conf = Config()


# todo 1.定义函数, 加载并处理原始数据集.
def load_raw_data(file_path):
    """
    从指定 JSON Lines 文件中加载数据, 处理为: '文件内容-标签索引'的元组列表, 供后续封装使用.
    :param file_path: 原始数据文件的路径（JSON Lines 格式，每行: {"text":"...", "intent":"..."}）
    :return: 列表嵌套元组, 例如: [('文本字符串', 标签整数索引), ('文本字符串', 标签整数索引), (...)]
    """
    # 1. 初始化结果列表, 存储处理后的数据.
    data = []
    # 2. 打开源文件.
    with open(file_path, encoding="utf-8") as f:
        # 3. 逐个读取每行 JSON 数据
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # 解析 JSON 行
                record = json.loads(line)
                text = record['text']
                intent = record['intent']
                # 将意图字符串转换为类别索引
                label = conf.class_list.index(intent)
                data.append((text, label))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # 跳过格式错误的行
                continue
    return data


# todo 2. 自定义数据集类(继承PyTorch中的DataSet)
class TextDataset(Dataset):
    # 1. 初始化函数
    def __init__(self, data_list):
        """
        初始化数据集, 接收原始数据列表, 将其转换为 DataLoader可以识别的格式.
        :param data_list: 列表嵌套元组, 例如: [('文本字符串', 3), ('文本字符串', 3), (...)]
        """
        # 1. 创建数据集对象.
        self.data_list = data_list

    # 2. 获取数据集大小
    def __len__(self):
        return len(self.data_list)

    # 3. 获取指定索引的数据
    def __getitem__(self, index):
        # 返回结果---> text: 文本字符串, label: 标签索引(整数形式)
        text, label = self.data_list[index]
        return text, label


# todo 3. 定义整理函数 -> 批量处理某一批次数据
def collate_fn(batch):
    """
    给DataLoader的一个批次(Batch)原始数据进行预处理: 分词, 填充, 转张量
    :param batch: 某一批次的数据, 例如: [(text1, label1), (text2, label2), ...]
    :return: 元组形式, 三个值分别是: input_ids , attention_mask, labels
        input_ids: 分词后token的ID, 形状为: (batch_size, max_length)
        attention_mask: 注意力掩码, 标记有效token和填充token, 形状和 input_ids一致.
        labels: 批次标签, 形状为: (batch_size,)
    """
    # 1. 提取文本和标签
    text = [item[0] for item in batch]
    label = [item[1] for item in batch]
    # 2. 调用BERT分词器的batch_encode_plus()方法, 对批量文本进行编码.
    """
    常用参数:
        add_special_tokens=True,            # 是否添加特殊标记(CLS, SEP, PAD, ...)
        padding='max_length',               # 填充策略
        max_length=conf.pad_size,           # 最大长度
        truncation=True,                    # 是否截断
        return_attention_mask=True,         # 是否返回注意力掩码
        return_tensors='pt'                 # 返回张量格式
    """
    output = conf.tokenizer(
        text,
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True, # 如果text的文本长度超过pad_size, 则截断
        return_attention_mask=True, # 对padding补齐部分进行掩码(掩码值设置为0)
        return_tensors='pt',
    )
    # 3. 获取分词结果, 获取input_ids和attention_mask
    input_ids = output['input_ids']
    attention_mask = output['attention_mask']
    # 4. 将labels转张量
    label = torch.tensor(label)
    # 5. 返回结果
    return input_ids, attention_mask, label


# todo 4. 构建数据加载器函数
def build_dataloader():
    """
    构建训练集, 验证集, 测试集的数据加载器(DataLoader)
    :return: 包含单个DataLoader的元素, 顺序为: (train_dataloader, dev_dataloader, test_dataloader)
    """
    # 1. 调用load_raw_data()函数, 加载原始数据 -> 训练集, 验证集, 测试集
    train_data = load_raw_data(conf.train_datapath)
    dev_data = load_raw_data(conf.dev_datapath)
    test_data = load_raw_data(conf.test_datapath)
    # 2. 将上述的数据(列表嵌套元组) -> 自定义数据集对象
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)
    # 3. 构建DataLoader: 指定批次大小, 是否打乱数据, 批量处理函数.
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    # 4. 返回三个数据集的dataloader结果
    return train_dataloader, dev_dataloader, test_dataloader
    pass


# todo 程序的主入口
if __name__ == '__main__':
    # # # 测试1: load_raw_data()函数
    # dev_data = load_raw_data(conf.dev_datapath)
    # print(dev_data)
    # # # 测试2: TextDataset()类
    # dev_dataset = TextDataset(dev_data)
    # print(f"验证集数据集的长度是:{len(dev_dataset)}")
    # dev_result = dev_dataset[0]
    # print(f"验证集中第一条数据是:{dev_result}")

    # 测试3: build_dataloader()函数
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # print("获取数据集加载器结果")
    # # 测试4: 获取数据集加载器对象.
    for batch in dev_dataloader:
        input_ids, attention_mask, labels = batch
        print(f"input_ids的形状: {input_ids.shape}")
        print(f"input_ids: {input_ids}")
        print("*"*100)
        print(f"attention_mask的形状: {attention_mask.shape}")
        print(f"attention_mask: {attention_mask}")
        print("*"*100)
        print(f"labels的形状: {labels.shape}")
        print(f"labels: {labels}")
        break
