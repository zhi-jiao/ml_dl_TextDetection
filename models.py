import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from utils import pad_sequence
import torch.optim as optim

# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.dropout(output)
        output = torch.mean(output, dim=1) 
        output = self.fc(output)
        return output

# GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.dropout(output)
        output = torch.mean(output, dim=1)
        output = self.fc(output)
        return output
# BiLSTM
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        output = self.dropout(output)
        output_avg = torch.mean(output, dim=1)  
        output = self.fc(output_avg)
        return output

# BiGRU
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bigru(embedded)
        output = self.dropout(output)
        output_avg = torch.mean(output, dim=1)  
        output = self.fc(output_avg)
        return output
    

# TextCNN

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]

        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len - filter_size + 1), ...]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [(batch_size, num_filters), ...]

        cat = torch.cat(pooled, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        output = self.fc(self.dropout(cat))  # [batch_size, output_dim]
        return output