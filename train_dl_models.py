from pytorchtools import EarlyStopping
from models import LSTMClassifier,GRUClassifier,BiLSTMClassifier,BiGRUClassifier,TextCNN
from datasets import Article_data
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import wandb
from utils import seed_torch
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
# set seed
from args import get_args

# Get arguments
model_args = get_args()

# Access parameters
vocab_size = model_args.vocab_size
embedding_dim = model_args.embedding_dim
DEVICE = model_args.device 
seed_torch(model_args.seed)
torch.backends.cudnn.enable =True

# parameters
output_dim = model_args.output_dim
vocab_size = model_args.vocab_size 
embedding_dim = model_args.embedding_dim
hidden_dim = model_args.hidden_dim
seq_length = model_args.seq_length
lr = model_args.lr
epoch = model_args.epoch
# textcnn
num_filters =model_args.num_filters
filter_sizes = model_args.filter_sizes  
dropout = model_args.dropout
patience = model_args.patience

train_path = model_args.train_path
test_path =  model_args.test_path
val_path = model_args.val_path
dic_path = model_args.dic_path

# load dataset
train_data = Article_data(train_path,dic_path,seq_length=seq_length)
val_data = Article_data(val_path,dic_path,seq_length=seq_length)
test_data = Article_data(test_path,dic_path,seq_length=seq_length)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
val_loader = DataLoader(val_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False) # 后续需要用于分析


if model_args.model =='GRU':
    model = GRUClassifier(vocab_size,embedding_dim,hidden_dim,output_dim)
if model_args.model =='LSTM':
    model = LSTMClassifier(vocab_size,embedding_dim,hidden_dim,output_dim)
if model_args.model =='BiGRU':
    model = BiGRUClassifier(vocab_size,embedding_dim,hidden_dim,output_dim)
if model_args.model =='BiLSTM':
    model = BiLSTMClassifier(vocab_size,embedding_dim,hidden_dim,output_dim)
if model_args.model =='TextCNN':
    model = TextCNN(vocab_size,embedding_dim,num_filters,filter_sizes,output_dim,0.5)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
early_stopping = EarlyStopping(patience, verbose=True,check_name=model_args.model + '.pt')
criterion = nn.CrossEntropyLoss()

model = model.to(DEVICE)
for i in range(epoch):
    model.train()
    print('train')
    train_loss = 0.0
    val_loss = 0.0
    for data,target in train_loader:
        target = F.one_hot(target.squeeze(), num_classes=2).to(torch.float)
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    print('Epoch ',i+1,' Train_loss: ',train_loss)
    print('val')
    model.eval()
    for data,target in val_loader:
        target = F.one_hot(target.squeeze(), num_classes=2).to(torch.float)
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        loss = criterion(output,target)
        val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    print('Epoch ',i+1,' Val_loss: ',val_loss)
    early_stopping(val_loss,model)
    
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Eval
model.eval()
print('Eval')
accuracy = 0.0
precision = 0.0
recall = 0.0
f1 = 0.0

out_list = []
out_csv = pd.DataFrame([])
for data,target in test_loader:
        target = F.one_hot(target.squeeze(), num_classes=2).to(torch.float)
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        output = output.detach().cpu().numpy()
        result = output.argmax(axis=-1)
        label = target.detach().cpu().numpy().argmax(axis=-1)
        
        accuracy += accuracy_score(label,result)
        precision += precision_score(label,result)
        recall  += recall_score(label,result)
        f1 += f1_score(label,result) 
        out_list.append(result)

accuracy = accuracy / len(test_loader)
precision = precision / len(test_loader)
recall = recall /  len(test_loader)
f1 = f1 / len(test_loader)
# out_csv['predict'] = out_list
# out_csv.to_csv('Bigru_out.csv',index=False)
print(f' Test Result: Accuracy:{accuracy:.6f}, Precision:{precision:.6f}, Recall:{recall:.6f}, F1-score : {f1:.6f}')        


