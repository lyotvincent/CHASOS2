"""
@description: Training Charid-anchor cnn model on PyTorch by Sun JiaLiang
@author: Sun JiaLiang
"""

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.scheduler import EarlyStopping
from source.parameters import PRETRAINED_DATA_PATH, CELL_LINE_LIST, CHROMOSOME_LIST
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from collections import defaultdict

def seed_torch(seed=9523):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU. torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False # type: ignore 
    torch.backends.cudnn.deterministic = True  # 选择确定性算法 # type: ignore

seed_torch()

epochs = 150
# patience = 20
patience = 160
learningrate=0.001
batch_size=128
dropout=0.7
nb_filter1=200
nb_filter2=100
nb_filter3=100
nb_filter4=64
filter_len1=19
filter_len2=11
filter_len3=11
filter_len4=7
pooling_size1=5
pooling_size2=5
pooling_size3=5
pooling_size4=2
gru=80
hidden=200
CHARID_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

monomer_dict = {'A': [1., 0., 0., 0.],
                'T': [0., 1., 0., 0.],
                'G': [0., 0., 1., 0.],
                'C': [0., 0., 0., 1.],
                'N': [0., 0., 0., 0.]}


def load_monomer_seqs(cell_line_list, split_mode):
    assert split_mode in ['random', 'chr', 'cell_line']
    monomer_train_data_dict = defaultdict(list)
    train_label_dict = defaultdict(list)

    def load_and_split_by_chr(file_path, label):
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                seq = fields[3].upper()
                monomers = [monomer_dict[i] for i in seq]
                monomer_train_data_dict[fields[0]].append(monomers)
                train_label_dict[fields[0]].append(label[0])

    split_mode_dict = {'chr':load_and_split_by_chr}
    for i, cell_line in enumerate(cell_line_list):
        print(f"loading {i} {cell_line}...")
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ocr.csv", [[1.]]) # type: ignore
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ccr.csv", [[1.]])
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ocr.csv", [[0.]])
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ccr.csv", [[0.]])
    return monomer_train_data_dict, train_label_dict

def load_monomer_encoded_data(cell_line_list, split_mode, val_chr_list=['chr22', 'chrX']):
    if split_mode == 'chr':
        charid_encoded_data_path = CHARID_MODEL_PATH+f'/encoded_data/{split_mode}'
        files_complete = True
        for cell_line in cell_line_list:
            for chr in CHROMOSOME_LIST:
                if os.path.exists(charid_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy') == False or\
                   os.path.exists(charid_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy') == False:
                    files_complete = False
                    break
            if files_complete == False:
                break
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array([]), np.array([]), np.array([]), np.array([])
        if files_complete:
            for cell_line in cell_line_list:
                for chr in CHROMOSOME_LIST:
                    feature_data = np.load(charid_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy')
                    labels = np.load(charid_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy')
                    print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
                    if len(feature_data) == 0: continue
                    if chr in val_chr_list:
                        if len(kmer_val_data_list) == 0:
                            kmer_val_data_list = feature_data
                        else:
                            kmer_val_data_list = np.concatenate((kmer_val_data_list, feature_data), axis=0)
                        if len(kmer_val_label_list) == 0:
                            kmer_val_label_list = labels
                        else:
                            kmer_val_label_list = np.concatenate((kmer_val_label_list, labels), axis=0)
                    else:
                        if len(kmer_train_data_list) == 0:
                            kmer_train_data_list = feature_data
                        else:
                            kmer_train_data_list = np.concatenate((kmer_train_data_list, feature_data), axis=0)
                        if len(kmer_train_label_list) == 0:
                            kmer_train_label_list = labels
                        else:
                            kmer_train_label_list = np.concatenate((kmer_train_label_list, labels), axis=0)
        else:
            for cell_line in cell_line_list:
                monomer_train_data_dict, train_label_dict = load_monomer_seqs([cell_line], split_mode=split_mode)
                for chr in CHROMOSOME_LIST:
                    print(f'{cell_line}_data_list_{chr}.npy & {cell_line}_labels_list_{chr}.npy are saving...')
                    np.save(charid_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy', np.array(monomer_train_data_dict[chr]))
                    np.save(charid_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy', np.array(train_label_dict[chr]))
    else:
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array([]), np.array([]), np.array([]), np.array([])
    return kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list

class PretrainedDataset(Dataset):
    def __init__(self, monomer_data, labels):
        self.monomer_data = torch.tensor(monomer_data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.monomer_data)

    def __getitem__(self, index):
            return self.monomer_data[index], self.labels[index]

class CharID_Anchor(nn.Module):
    '''
    @thop.profile: macs=64275400.0, params=492945.0
    @model parameter number: 595985
    '''
    def __init__(self):
        super().__init__()

        self.sequential_1 = nn.Sequential(
            nn.Conv1d(4, nb_filter1, filter_len1, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size1),
            nn.Conv1d(nb_filter1, nb_filter2, filter_len2, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size2),
            nn.Conv1d(nb_filter2, nb_filter3, filter_len3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size3),
            nn.Conv1d(nb_filter3, nb_filter4, filter_len4, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size4),
        )
        self.gru = nn.GRU(64, gru, bidirectional=True)
        self.att = nn.MultiheadAttention(gru*2, 1, dropout=dropout)
        self.sequential_2 = nn.Sequential(
            nn.Linear(gru*2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # sequence_input = Input(shape=(1000,4))
        # input = input.reshape(-1, 1000, 4)
        input = input.transpose(1,2)
        # input = input.reshape(-1, 1000, 1, 4)
        # input = input.to(memory_format=torch.channels_last)
        h = self.sequential_1(input)
        h = h.transpose(1,2)
        h = self.gru(h)[0]
        at = self.att(h, h, h)[0]
        h = torch.mul(h, at)
        h = torch.sum(h, dim=1)
        h = self.sequential_2(h)
        return h

def compute_loss(pred_scores, labels):
    loss_a = nn.BCELoss()(pred_scores, labels)
    loss = loss_a
    return loss

def compute_one_acc(pred1, label1):
    """计算精度
    Args:
        pred_scores (torch.Tensor): 预测分数，形状为 [batch_size, 1]
        labels (torch.Tensor): 真实标签，形状为 [batch_size, 1]
    Returns:
        float: 精度值(0到1之间的浮点数)
    """
    # 将预测分数四舍五入为0或1，表示预测标签
    pred1 = torch.round(pred1)
    # 计算预测标签与真实标签相同的数量
    correct = (pred1 == label1).sum().item()
    # 计算精度值
    accuracy = correct / len(label1)
    return accuracy

'''
# see FLOPs and parameters
model = CharID_Anchor()

from thop import profile
input = torch.randn(1, 1000, 4)
macs, params = profile(model, inputs=(input, ))
print(macs, params)

# from torchstat import stat
# stat(model, (1000, 4))

print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
exit()
'''

# * prepare log
start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
log_out = open(CHARID_MODEL_PATH+r'/logs/log_'+log_time+'.txt', 'w', buffering=8)
train_writer = SummaryWriter(CHARID_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/train')
val_writer = SummaryWriter(CHARID_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val')

# * data loader
SPLIT_MODE = 'chr'
VAL_CHR_LIST = ['chr22', 'chrX']
monomer_train_data_list, train_label_list, monomer_val_data_list, val_label_list = load_monomer_encoded_data(["GM12878"], SPLIT_MODE, VAL_CHR_LIST)
print("="*7)
log_info = f"""len(kmer_train_data_list): {monomer_train_data_list.shape}\nlen(train_label_list): {train_label_list.shape}\nlen(kmer_val_data_list): {monomer_val_data_list.shape}\nlen(val_label_list): {val_label_list.shape}"""
print(log_info)
log_out.write(log_info+"\n")

train_dataset = PretrainedDataset(monomer_train_data_list, train_label_list)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_dataset = PretrainedDataset(monomer_val_data_list, val_label_list)
val_dataloader = DataLoader(val_dataset, batch_size=128, pin_memory=True)
print("="*7)
print(f"len(train_dataloader)==train_batch_num: {len(train_dataloader)}")
print(f"len(val_dataloader)==val_batch_num: {len(val_dataloader)}")


# * model contructor
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
else:
    device = torch.device('cpu')
model = CharID_Anchor()
model.to(device)

# write model info
log_out.write(str(model)+'\n')
# write model parameter number
log_info = f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
log_out.write(log_info+"\n")
print(log_info) # model parameter number: 595985

# * scheduler
adagrad_new = torch.optim.Adagrad(model.parameters(), lr=float(learningrate))
early_stopping = EarlyStopping(mode='min', patience=patience, verbose=True)

# best metrics record
best_val_loss = 1e10
best_val_loss_log_info = None
best_val_anchor_acc = 0
best_val_anchor_acc_log_info = None
best_val_ocr_acc = 0
best_val_ocr_acc_log_info = None

# model.compile(loss='binary_crossentropy',optimizer=adagrad_new,metrics=['accuracy'])


# * model train
for e in range(epochs):
    print(f"Epoch {e}")
    log_out.write(f"Epoch {e}\n")
    print('----------')
    log_out.write('----------\n')

    for batch_num, batch in enumerate(train_dataloader):
        model.train()
        # batch_kmer_data, batch_mono_data, batch_labels = batch
        # batch_kmer_data, batch_mono_data, batch_labels = batch_kmer_data.to(device), batch_mono_data.to(device), batch_labels.to(device)
        batch_kmer_data, batch_labels = batch
        batch_kmer_data, batch_labels = batch_kmer_data.to(device), batch_labels.to(device)
        # 梯度清零
        adagrad_new.zero_grad()
        # 计算模型的输出
        pred_scores = model(batch_kmer_data)
        # 计算损失函数
        loss = compute_loss(pred_scores, batch_labels)
        # 反向传播
        loss.backward()
        # 更新梯度
        adagrad_new.step()

        if batch_num % 100 == 0:
            model.eval()
            with torch.no_grad():
                anchor_acc = compute_one_acc(pred_scores, batch_labels)
                val_loss, val_anchor_acc = 0, 0
                val_batch_num = len(val_dataloader)
                for val_batch in val_dataloader:
                    # print(batch_num, 9)
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_mono_data.to(device), val_batch_labels.to(device)
                    val_batch_kmer_data, val_batch_labels = val_batch
                    val_batch_kmer_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_labels.to(device)
                    val_pred_scores = model(val_batch_kmer_data)
                    val_loss += compute_loss(val_pred_scores, val_batch_labels).item()
                    temp_val_anchor_acc = compute_one_acc(val_pred_scores, val_batch_labels)
                    val_anchor_acc += temp_val_anchor_acc
                val_loss, val_anchor_acc = val_loss/val_batch_num, val_anchor_acc/val_batch_num
                log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, anchor_acc: {anchor_acc:.4f}, val_loss: {val_loss:.4f}, val_anchor_acc: {val_anchor_acc:.4f}"
                print(log_info)
                log_out.write(log_info + '\n')
                train_writer.add_scalar(tag='loss', scalar_value=loss, global_step=e*len(train_dataloader)+batch_num)
                train_writer.add_scalar(tag='anchor_acc', scalar_value=anchor_acc, global_step=e*len(train_dataloader)+batch_num)
                val_writer.add_scalar(tag='loss', scalar_value=val_loss, global_step=e*len(train_dataloader)+batch_num)
                val_writer.add_scalar(tag='anchor_acc', scalar_value=val_anchor_acc, global_step=e*len(train_dataloader)+batch_num)
                # val_writer.add_pr_curve(tag='pr_curve', labels=labels, predictions=predictions, global_step=0)
                if val_loss < best_val_loss: # update best val loss
                    best_val_loss = val_loss
                    best_val_loss_log_info = f"Epoch {e}\t"+log_info
                if (val_anchor_acc > best_val_anchor_acc): # update best val anchor acc
                    best_val_anchor_acc = val_anchor_acc
                    best_val_anchor_acc_log_info = f"Epoch {e}\t"+log_info
            # 早停止
            early_stopping(metric=val_loss, current_epoch=e)
        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            log_info = f"Early stopping at epoch {e}, batch {batch_num}"
            print(log_info)
            log_out.write(f'{log_info}\n')
            break #跳出迭代，结束训练
    if early_stopping.early_stop:
        break


train_writer.close()
val_writer.close()

print('----------')
log_out.write('----------\n')
print(f"Best val loss record: {best_val_loss_log_info}")
log_out.write(f"Best val loss record: {best_val_loss_log_info}\n")
print(f"Best val anchor acc record: {best_val_anchor_acc_log_info}")
log_out.write(f"Best val anchor acc record: {best_val_anchor_acc_log_info}\n")

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()

