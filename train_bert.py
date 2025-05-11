import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from pytorchtools import EarlyStopping
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from utils import seed_torch
import pandas as pd
from args import get_args

# Get arguments
model_args = get_args()

seed_torch(model_args.seed)
DEVICE = model_args.patience
epoch =  model_args.epoch

early_stopping = EarlyStopping(model_args.patience, verbose=True,check_name=model_args + '.pt')

def preprocess_text(input):
    input = eval(input)
    sentence = ' '.join(input)
    return sentence

train_path = model_args.train_path
test_path =  model_args.test_path
val_path = model_args.val_path


train_csv = pd.read_csv(train_path)
test_csv = pd.read_csv(test_path)
val_csv = pd.read_csv(val_path)

# 加载模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


def get_dataloader(csv_data,shuffle=True):
    corpus  = list(csv_data['text'].apply(preprocess_text))
    inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')
    labels = torch.LongTensor(csv_data['label'])
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=shuffle)
    return dataloader


# data_loader
train_loader = get_dataloader(train_csv,True)
test_loader = get_dataloader(test_csv,False)
val_loader = get_dataloader(val_csv,True)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(DEVICE)
for i in range(epoch):
    model.train()
    print('train')
    train_loss = 0.0
    val_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, target = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=target)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    print('Epoch ',i+1,' Train_loss: ',train_loss)
    print('val')
    model.eval()
    for batch in val_loader:
        input_ids, attention_mask, target = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        target = target.to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=target)
        loss = outputs.loss
        logits = outputs.logits
        val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    print('Epoch ',i+1,' Val_loss: ',val_loss)
    early_stopping(val_loss,model)
    
    # wandb.log({'train loss':train_loss,'val_loss':val_loss})
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
result_list = []
output_csv = pd.DataFrame([])
for batch in test_loader:
        input_ids, attention_mask, target = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        target = target.to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=target)
        logits = outputs.logits
        output = logits.detach().cpu().numpy()
        result = output.argmax(axis=-1)
        label = target.detach().cpu().numpy()
        # 计算评价标准
        accuracy += accuracy_score(label,result)
        precision += precision_score(label,result)
        recall  += recall_score(label,result)
        f1 += f1_score(label,result) 
        result_list.append(result)

accuracy = accuracy / len(test_loader)
precision = precision / len(test_loader)
recall = recall /  len(test_loader)
f1 = f1 / len(test_loader)
output_csv['predict'] = result_list
output_csv.to_csv('./output/code/output_bert.csv',index=False)              
print(f' Test Result: Accuracy:{accuracy:.6f}, Precision:{precision:.6f}, Recall:{recall:.6f}, F1-score : {f1:.6f}')        
