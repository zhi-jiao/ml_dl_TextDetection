import torch
import pickle
from torch.utils.data import Dataset
from utils import pad_sequence,text_to_int,text_to_int_zh
import pandas as pd
import numpy as np

class Article_data(Dataset):
    def __init__(self,data_file,dic_file,seq_length =128):
        self.data = pd.read_csv(data_file)
        with open(dic_file, 'rb') as f:
            self.token2id =pickle.load(f)
        self.seq_length = seq_length
        self.target = torch.LongTensor(self.data['label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        target = self.target[index]
        sentence = self.data['text'][index]
        # tokens = torch.LongTensor(text_to_int(sentence,self.token2id))
        tokens = torch.LongTensor(text_to_int_zh(sentence,self.token2id))
        tokens = pad_sequence(tokens,self.seq_length)
        return tokens,target        

if __name__ == '__main__':
    data_file = './data/true_data/samples.csv'  # 替换为你的数据文件路径
    dic_file = './data/true_data/token2id.pkl'  # 替换为你的词典文件路径
    dataset = Article_data(data_file, dic_file, seq_length=128)
    print(f"数据集大小: {len(dataset)}")
    tokens, target = dataset[0]
    print(f"Tokens: {tokens}, Target: {target}")