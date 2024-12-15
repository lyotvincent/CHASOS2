
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from source.parameters import POS_LOOP_PATH, NEG_LOOP_PATH, CELL_LINE_LIST, CHROMOSOME_LIST
from source.scheduler import EarlyStopping

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
np.random.seed(12345)


DEEPCTCFLOOP_PATH = os.path.dirname(os.path.abspath(__file__))
params = {'batch_size': 4, 'dense_unit': 112, 'drop_out_cnn': 0.4279770967884926, 'drop_out_lstm': 0.05028428952624636, 'filter': 208, 'kernel_initializer': 'random_uniform', 'l2_reg': 5.2164660610264974e-05, 'learning_rate': 0.00010199140620075788, 'lstm_unit': 64, 'pool_size': 4, 'window_size': 13}

'''Build the DeepLncCTCF model'''
class DeepLncCTCFModel(nn.Module):
    def __init__(self):
        super(DeepLncCTCFModel, self).__init__()
        
        self.conv1 = nn.Conv1d(4, params['filter'], params['window_size'])
        self.maxpool = nn.MaxPool1d(params['pool_size'], stride=params['pool_size'])
        self.conv2 = nn.Conv1d(params['filter'], params['filter'], params['window_size'])
        self.dropout1 = nn.Dropout(params['drop_out_cnn'])
        self.dropout2 = nn.Dropout(params['drop_out_cnn'])
        self.lstm = nn.LSTM(params['filter'], params['lstm_unit'], 
                            bidirectional=True, batch_first=True)
        self.attention_w = nn.Linear(61, 61)
        self.dropout3 = nn.Dropout(params['drop_out_lstm'])
        self.dense = nn.Linear(params['lstm_unit']*2, params['dense_unit'])
        self.output = nn.Linear(params['dense_unit'], 1)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, 1038, 4) (B,L,C)
        inputs = inputs.transpose(1, 2) # (batch_size, 4, 1038) (B,C,L)
        cnn_out = F.relu(self.conv1(inputs)) # (batch_size, 208, 1026) (B,C,L)
        pooling_out = self.maxpool(cnn_out) # (batch_size, 208, 256) (B,C,L)
        dropout1 = self.dropout1(pooling_out)
        cnn_out2 = F.relu(self.conv2(dropout1)) # (batch_size, 208, 244) (B,C,L)
        pooling_out2 = self.maxpool(cnn_out2) # (batch_size, 208, 61) (B,C,L)
        dropout2 = self.dropout2(pooling_out2)
        lstm_out, _ = self.lstm(dropout2.transpose(1, 2)) # (batch_size, 61, 128) (B,L,C)
        attention_w = torch.softmax(self.attention_w(lstm_out.transpose(1, 2)), dim=2) # (batch_size, 128, 61) (B,C,L)
        attention_w = attention_w.transpose(1, 2) # (batch_size, 61, 128) (B,L,C)
        attention_out = torch.mul(lstm_out, attention_w) # (batch_size, 61, 128) (B,L,C)
        attention_out = torch.sum(attention_out, dim=1) # (batch_size, 128) (B,C)
        dropout3 = self.dropout3(attention_out)
        dense_out = F.relu(self.dense(dropout3))
        output = torch.sigmoid(self.output(dense_out))
        return output


monomer_dict = {'A': [1., 0., 0., 0.],
                'T': [0., 1., 0., 0.],
                'G': [0., 0., 1., 0.],
                'C': [0., 0., 0., 1.],
                'N': [0., 0., 0., 0.]}


def gen_data(cell_line_list):
    '''
    生成环的训练数据
    '''
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        seqs_list = dict()
        labels_list = dict()
        for chr in CHROMOSOME_LIST:
            seqs_list[chr] = list()
            labels_list[chr] = list()
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            print("=。=", pos_loop_path)
            f = open(pos_loop_path, 'r')
            lines = f.readlines()
            f.close()
            pos_total_sample_num += len(lines)
            for line in lines:
                fields = line.strip().split()
                chrom = fields[0]
                # seq feature
                seq1 = fields[5][240:759]
                seq2 = fields[11][240:759]
                if "N" in seq1 or "N" in seq2:
                    continue
                seq = seq1 + seq2
                seq = [monomer_dict[i] for i in seq]
                seqs_list[chrom].append(seq)
                labels_list[chrom].append(1.)
        neg_sample_num = pos_total_sample_num // 10
        for typee in range(1, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                print("=。=", neg_loop_path)
                f = open(neg_loop_path, 'r')
                lines = f.readlines()
                f.close()
                lines = lines[:neg_sample_num]
                for line in lines:
                    fields = line.strip().split()
                    chrom = fields[0]
                    # seq feature
                    seq1 = fields[5][240:759]
                    seq2 = fields[11][240:759]
                    if "N" in seq1 or "N" in seq2:
                        continue
                    seq = seq1 + seq2
                    seq = [monomer_dict[i] for i in seq]
                    seqs_list[chrom].append(seq)
                    labels_list[chrom].append(0.)
        for chr in CHROMOSOME_LIST:
            np.save(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_data_list_{chr}.npy', np.array(seqs_list[chr]))
            np.save(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_labels_list_{chr}.npy', np.array(labels_list[chr]))

def load_data(cell_line_list, val_chr_list):
    train_seqs = np.array([])
    train_labels = np.array([])
    val_seqs = np.array([])
    val_labels = np.array([])
    for cell_line in cell_line_list:
        for chr in CHROMOSOME_LIST:
            feature_data = np.load(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_data_list_{chr}.npy')
            labels = np.load(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_labels_list_{chr}.npy')
            print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
            if len(feature_data) == 0: continue
            if chr in val_chr_list:
                if len(val_seqs) == 0:
                    val_seqs = feature_data
                    val_labels = labels
                else:
                    val_seqs = np.concatenate((val_seqs, feature_data), axis=0)
                    val_labels = np.concatenate((val_labels, labels), axis=0)
            else:
                if len(train_seqs) == 0:
                    train_seqs = feature_data
                    train_labels = labels
                else:
                    train_seqs = np.concatenate((train_seqs, feature_data), axis=0)
                    train_labels = np.concatenate((train_labels, labels), axis=0)
    return train_seqs, train_labels, val_seqs, val_labels

def load_data_by_health(cell_line_list, val_cell_line_list):
    train_seqs = np.array([])
    train_labels = np.array([])
    val_seqs = np.array([])
    val_labels = np.array([])
    for cell_line in cell_line_list:
        for chr in CHROMOSOME_LIST:
            feature_data = np.load(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_data_list_{chr}.npy')
            labels = np.load(DEEPCTCFLOOP_PATH+f'/data/{cell_line}_labels_list_{chr}.npy')
            print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
            if len(feature_data) == 0: continue
            if cell_line in val_cell_line_list:
                if len(val_seqs) == 0:
                    val_seqs = feature_data
                    val_labels = labels
                else:
                    val_seqs = np.concatenate((val_seqs, feature_data), axis=0)
                    val_labels = np.concatenate((val_labels, labels), axis=0)
            else:
                if len(train_seqs) == 0:
                    train_seqs = feature_data
                    train_labels = labels
                else:
                    train_seqs = np.concatenate((train_seqs, feature_data), axis=0)
                    train_labels = np.concatenate((train_labels, labels), axis=0)
    return train_seqs, train_labels, val_seqs, val_labels


class DeepCTCFLoopDataset(Dataset):
    def __init__(self, kmer_data, labels):
        self.kmer_data = torch.tensor(kmer_data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.kmer_data)

    def __getitem__(self, index):
        return self.kmer_data[index], self.labels[index]

def compute_loop_acc(pred1, label1):
    if len(pred1.shape) == 2:
        pred1 = pred1.squeeze(1)
    # 将预测分数四舍五入为0或1，表示预测标签
    pred1 = torch.round(pred1)
    # print(pred1)
    # print(label1)
    # 计算预测标签与真实标签相同的数量
    correct = (pred1 == label1).sum().item()
    # print(correct, len(label1))
    # 计算精度值
    accuracy = correct / len(label1)
    return accuracy


def train():
    # * prepare log
    start_time = time.time()
    log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_out = open(DEEPCTCFLOOP_PATH+r'/logs/log_'+log_time+'.txt', 'w', buffering=8)
    train_writer = SummaryWriter(DEEPCTCFLOOP_PATH+f'/logs/tensorboard_log_{log_time}/train')
    val_writer = SummaryWriter(DEEPCTCFLOOP_PATH+f'/logs/tensorboard_log_{log_time}/val')

    # * hyper parameters
    SPLIT_MODE = 'chr'
    VAL_CHR_LIST = ['chr22', 'chrX']
    BATCH_SIZE = params['batch_size']
    LOG_STEP = 100
    EPOCH = 80
    log_out.write(f"SPLIT_MODE: {SPLIT_MODE}\n")
    log_out.write(f"VAL_CHR_LIST: {VAL_CHR_LIST}\n")
    log_out.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    log_out.write(f"LOG_STEP: {LOG_STEP}\n")
    log_out.write(f"MAX Epoch num: {EPOCH}\n")

    # * data loader
    cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
    val_cell_line_list = ["Caco-2", "HCT116", "K562", "MCF-7"]
    log_out.write(f"cell_line_list: {cell_line_list}\n")
    # kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list, monomer_train_data_list, monomer_val_data_list = load_kmer_encoded_data(SPLIT_MODE, STRIDE)
    train_features,  train_labels, val_features, val_labels = load_data(cell_line_list=cell_line_list, val_chr_list=VAL_CHR_LIST)
    # train_features,  train_labels, val_features, val_labels = load_data_by_health(cell_line_list=cell_line_list, val_cell_line_list=val_cell_line_list)
    print("="*7)
    log_info = f"""len(train_features): {train_features.shape}
    len(train_labels): {train_labels.shape}
    len(val_features): {val_features.shape}
    len(val_labels): {val_labels.shape}"""
    print(log_info)
    log_out.write(log_info+"\n")

    train_dataset = DeepCTCFLoopDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    val_dataset = DeepCTCFLoopDataset(val_features, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    print("="*7)
    print(f"len(train_dataloader)==train_batch_num: {len(train_dataloader)}")
    print(f"len(val_dataloader)==val_batch_num: {len(val_dataloader)}")


    # * model contructor
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log_info = f"""device: {device}"""
    print(log_info)
    log_out.write(log_info+"\n")
    model = DeepLncCTCFModel()
    model.to(device)


    # * scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], eps=10**-8, weight_decay=0.01)
    log_out.write(f"optimizer: {type(optimizer)}\n")
    early_stopping = EarlyStopping(mode="min", patience=200, verbose=True, delta=0, start_epoch=0, log_file=log_out)

    # best metrics record
    best_val_loss = 1e10
    best_val_loss_log_info = None
    best_val_acc = 0
    best_val_acc_log_info = None

    # * model train
    for e in range(EPOCH):
        print(f"Epoch {e}")
        log_out.write(f"Epoch {e}\n")
        print('----------')
        log_out.write('----------\n')

        for batch_num, batch in enumerate(train_dataloader):
            model.train()
            batch_features, batch_labels = batch
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 计算模型的输出
            pred_scores = model(batch_features)
            # 计算损失函数
            loss = nn.BCELoss()(pred_scores.squeeze(1), batch_labels)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()

            if batch_num % LOG_STEP == 0:
                model.eval()
                with torch.no_grad():
                    acc = compute_loop_acc(pred_scores, batch_labels)
                    val_loss, val_acc = 0, 0
                    val_batch_num = len(val_dataloader)
                    for val_batch in val_dataloader:
                        val_batch_features, val_batch_labels = val_batch
                        val_batch_features, val_batch_labels = val_batch_features.to(device), val_batch_labels.to(device)
                        val_pred_scores = model(val_batch_features)
                        temp_val_loss = nn.BCELoss()(val_pred_scores.squeeze(1), val_batch_labels)
                        val_loss += temp_val_loss.item()
                        val_acc += compute_loop_acc(val_pred_scores, val_batch_labels)
                    val_loss, val_acc = val_loss/val_batch_num, val_acc/val_batch_num
                    log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                    print(log_info)
                    log_out.write(log_info + '\n')
                    global_step = e*len(train_dataloader)+batch_num
                    train_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=global_step)
                    train_writer.add_scalar(tag='acc', scalar_value=acc, global_step=global_step)
                    val_writer.add_scalar(tag='loss', scalar_value=val_loss, global_step=global_step)
                    val_writer.add_scalar(tag='acc', scalar_value=val_acc, global_step=global_step)
                    if val_loss < best_val_loss: # update best val loss
                        best_val_loss = val_loss
                        best_val_loss_log_info = f"Epoch {e}\t"+log_info
                        checkpoint = {
                            'epoch':e,
                            'model':model.state_dict()
                        }
                        torch.save(checkpoint, DEEPCTCFLOOP_PATH+f'/model/LoopModel_{log_time}.pt')
                    if val_acc > best_val_acc: # update best val acc
                        best_val_acc = val_acc
                        best_val_acc_log_info = f"Epoch {e}\t"+log_info
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
    print(f"Best val_loss record: {best_val_loss_log_info}")
    log_out.write(f"Best val_loss record: {best_val_loss_log_info}\n")
    print(f"Best val_acc record: {best_val_acc_log_info}")
    log_out.write(f"Best val_acc record: {best_val_acc_log_info}\n")

    run_time = time.time() - start_time
    print(f'Run time: {run_time}s')
    log_out.write(f"Run time: {run_time}s\n")
    log_out.close()



if __name__ == '__main__':
    # gen_data(cell_line_list=["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"])

    # load_data(cell_line_list=["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"], val_chr_list=['chr22', 'chrX'])

    train()
