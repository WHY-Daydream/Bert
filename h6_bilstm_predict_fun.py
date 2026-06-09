# 该 .py文件的作用是 -> 预测分类的 函数版, 对接后续的 API 和 APP版.


# 导包.
from config import Config
import torch
from h3_bilstm_classifier_model import BiLSTMClassifier
import time
# 压制警告.
import warnings
warnings.filterwarnings("ignore")


# todo 1.加载全局配置.
conf = Config()

# todo 2. 准备BiLSTM学生模型（模块级变量，供 predict_fun 使用）
model = None  # 由 init_model() 初始化


def init_model(model_type='soft'):
    """
    加载指定类型的蒸馏模型。
    :param model_type: 'soft' 加载软标签蒸馏模型, 'hard' 加载硬标签蒸馏模型
    """
    global model
    model = BiLSTMClassifier()

    model_paths = {
        'hard': conf.bert_model_distill_model_path_hard,
        'soft': conf.bert_model_distill_model_path_soft,
    }

    path = model_paths.get(model_type)
    if path is None:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(model_paths.keys())}")

    print(f"[init_model] 加载 {model_type} 标签蒸馏模型: {path}")
    # 如果是gpu训练的模型, 用cpu设备加载会有错误, 设定map_location=conf.device指定设备解决此问题
    # weights_only=True 限制 pickle 只能解析基础数值 / 张量，防止加载被篡改的 pth 文件导致代码注入、恶意脚本执行，PyTorch 官方推荐生产环境固定 weights_only=True。
    model.load_state_dict(torch.load(path, map_location=conf.device, weights_only=True))

    # 模型放到设备上
    model.to(conf.device)

    # 设置模型为评估模式.
    model.eval()


# 模块导入时默认加载软标签蒸馏模型（保持向后兼容）
init_model('soft')
# todo 3. 定义预测函数, 接收文本数据, 返回分类结果.
def predict_fun(data_dict):
    """
    接收包含文本的字典, 通过BERT模型预测文本类别, 返回带预测结果的字典.
    :param data_dict: 输入字典, 格式为: {'text': '待预测文本内容'}
    :return:  {'text': '待预测文本内容', 'pred_class': '文本类别'}
    """
    # 1. 提取输入文本, 获取待预测的字符串.

    # 2. 文本编码, 将原始文本 -> BERT模型可识别的token id
    """
    常用参数: 
        add_special_tokens=True,            # 是否添加特殊标记(CLS, SEP, PAD, ...)
        padding='max_length',               # 填充策略
        max_length=conf.pad_size,           # 最大长度
        truncation=True,                    # 是否截断
        return_attention_mask=True,         # 是否返回注意力掩码
        return_tensors='pt'                 # 返回张量格式
    """
    start_time = time.time()
    output = conf.tokenizer(
        data_dict['text'],
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,  # 最大长度
        truncation=True,  # 是否截断
        return_attention_mask=True,  # 是否返回注意力掩码
        return_tensors='pt'  # 返回张量格式
    )

    # 3. 提取模型所需要的特征.
    input_ids = output['input_ids']
    attention_mask = output['attention_mask']

    # 4. 转换数据并指定设备.
    input_ids = input_ids.to(conf.device)
    attention_mask = attention_mask.to(conf.device)

    # 5. 模型预测动作. 禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 5.1 前向传播
        logits = model(input_ids, attention_mask)
        # 5.2 获取预测类别索引.
        y_label = torch.argmax(logits, dim=-1) #
        # 5.3 转换索引格式, 从PyTorch张量  -> Python的标量.
        print(y_label)  # tensor([8], device='cuda:0')
        print(type(y_label)) #  # <class 'torch.Tensor'>

        # 5.4 获取预测类别. 根据索引 -> 类别名
        y_label = y_label.item()
        # 5.5 打印结果
        y_label_names = conf.class_list[y_label]
        # 5.6 添加预测结果到字典, 并返回.
        data_dict['pred_class'] = y_label_names
    # 6. 返回结果
    data_dict['cost_time'] = (time.time() - start_time) * 1000
    return data_dict
    pass

# todo 4. 测试代码.
if __name__ == '__main__':
    # 1. 创建测试数据集.
    data_dict = {'text': '体验2D巅峰 倚天屠龙记十大创新概览'}
    # 2. 调用预测接口, 并打印结果.
    print(predict_fun(data_dict))
