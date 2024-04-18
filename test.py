import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, roc_auc_score
def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [i for i in data_columns]
    data_frame.columns = data_columns
    trainlabel = data_frame[data_frame.columns[0]]
    proBert_seq = data_frame[data_frame.columns[1]]
    return  np.array(trainlabel), np.array(proBert_seq)
device = torch.device("cuda:0")

class AQYDataset(Dataset):
    def __init__(self, df, label, device):
        self.protein_seq = df
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        label = self.label_list[index]
        return sample, label

    def __len__(self):
        return len(self.protein_seq)
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
        query = self.query(x)
        # print(query.shape)
        key = self.key(x)
        # print(key.shape)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)
        output = torch.matmul(attention, value)
        return output
class ESM_CNN_Attention(nn.Module):
    def __init__(self, embedding_dim=50, hidden_dim=32, n_layers=1):
        super(ESM_CNN_Attention, self).__init__()
        self.bert = AutoModel.from_pretrained("/home/hd/SGao/new/esm2")
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        out_channel = 32
        self.conv1 = nn.Conv1d(1280, 512, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv1d(512, 128, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding='same')
        self.batch1 = nn.BatchNorm1d(512)
        self.batch2 = nn.BatchNorm1d(128)
        self.batch3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.self_attention = SelfAttention(out_channel)
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        pooled_output, hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled_output = self.dropout(pooled_output)

        imput = pooled_output.permute(0, 2, 1)
        conv1_output = self.conv1(imput)
        batch1_output = self.batch1(conv1_output)
        conv2_output = self.conv2(batch1_output)
        batch2_output = self.batch2(conv2_output)
        conv3_output = self.conv3(batch2_output)
        batch3_output = self.batch3(conv3_output)
        prot_out = torch.mean(batch3_output, axis=2, keepdim=True)
        prot_out = prot_out.permute(0, 2, 1)
        # prot_out = prot_out.squeeze(1)
        attn_output = self.self_attention(prot_out)
        out2 = self.fc1(attn_output)
        # out2 = self.dropout1(out2)
        logit = self.fc2(out2)

        return nn.Sigmoid()(logit)

# load model
model = ESM_CNN_Attention().to(device)
model.load_state_dict(torch.load('num_model4_acc0.9427083333333334.pkl'))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("data_path_esm2")
# load data
test_data_path = pd.read_csv("testdata_path") 
test_Y, test_X = tokenize(test_data_path)
test_dataset = AQYDataset(test_X, test_Y, device)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# test model
pred_list = []
label_list = []
for samples, labels in test_loader:
    input_ids = samples['input_ids'].to(device)
    attention_mask = samples['attention_mask'].to(device)
    with torch.no_grad():
        preds = model(input_ids, attention_mask)
    pred_list.extend(preds.squeeze().cpu().detach().numpy())
    label_list.extend(labels.squeeze().cpu().detach().numpy())

# caculate
def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    return SN, SP, ACC, MCC


def cal_score(pred, label):
    AUC = roc_auc_score(list(label), pred)
    pred = np.around(pred)
    label = np.array(label)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC = Model_Evaluate(cm)
    print("Model score --- SN:{0:.3f}  SP:{1:.3f}  ACC:{2:.3f}  MCC:{3:.3f}  }".format(SN, SP, ACC, MCC))
    return SN, SP, ACC, MCC, cm

SN, SP, ACC, MCC, cm = cal_score(pred_list, label_list)
print("Confusion Matrix:")
print(cm)
print("Sensitivity:", SN)
print("Specificity:", SP)
print("Accuracy:", ACC)
print("MCC:", MCC)


