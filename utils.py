import string
import re 
import json
import torch
import random
import os
import numpy as np 
import jieba

# set seed
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# article data preprocessed
def preprocess_article_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\n+", " ", text) 
    text = re.sub(r"<.*?>", "", text)

    try:
        json_data = json.loads(text)
        text = json.dumps(json_data)
    except json.JSONDecodeError:
        pass
    return text

# pad_sequence

def pad_sequence(input_tensor,seq_length):
    """
    将输入的一维张量进行pad处理，使得长度为128
    """
    pad_size = seq_length - input_tensor.size(0)
    if pad_size > 0:
        padding = torch.zeros(pad_size, dtype=torch.long)
        padded_tensor = torch.cat((input_tensor, padding))
    else:
        padded_tensor = input_tensor[:seq_length]
    return padded_tensor

# text to int
# input text: list
def text_to_int(text, token2id):
    tokens = eval(text)
    int_list = [token2id[token] for token in tokens if token in token2id]
    return int_list

def text_to_int_zh(sentence, token2id):
    tokens = jieba.cut(sentence)
    return [token2id[token] for token in tokens if token in token2id]