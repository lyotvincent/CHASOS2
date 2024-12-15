'''
@author: 孙嘉良
@description: use ml data train loop model
'''

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained_model'))

import random
import numpy as np
import pandas as pd
from parameters import LOOP_MODEL_PATH
from preprocess_data import load_loop_dl_data
from dl_loop_models import *
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from loop_dl_metrics import compute_loop_acc
from scheduler import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

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

# * prepare log
start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
log_out = open(LOOP_MODEL_PATH+r'/logs/log_'+log_time+'.txt', 'w', buffering=8)
train_writer = SummaryWriter(LOOP_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/train')
val_writer = SummaryWriter(LOOP_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val')

# * hyper parameters
SPLIT_MODE = 'chr'
VAL_CHR_LIST = ['chr22', 'chrX']
STRIDE = 1
INIT_BATCH_SIZE = 128 # 使用全部cell line 做chr val的
INIT_BATCH_SIZE = 16 # 使用GM12878 做chr val的
BATCH_SIZE = 64
LOG_STEP = int(100*(INIT_BATCH_SIZE/BATCH_SIZE))
LOG_STEP = 16
EPOCH = 1000
LEARNING_RATE = 1E-2
PATIENCE = 80
DELTA = 5E-5
START_EPOCH = 0
log_out.write(f"SPLIT_MODE: {SPLIT_MODE}\n")
log_out.write(f"VAL_CHR_LIST: {VAL_CHR_LIST}\n")
log_out.write(f"STRIDE: {STRIDE}\n")
log_out.write(f"INIT_BATCH_SIZE: {INIT_BATCH_SIZE}\n")
log_out.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
log_out.write(f"LOG_STEP: {LOG_STEP}\n")
log_out.write(f"MAX Epoch num: {EPOCH}\n")
log_out.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
log_out.write(f"PATIENCE: {PATIENCE}\n")
log_out.write(f"DELTA: {DELTA}\n")
log_out.write(f"START_EPOCH: {START_EPOCH}\n")

# * data loader
data = pd.read_table(LOOP_MODEL_PATH+"/ml_training_data/GM12878_val_loss_2023_05_26_21_59_09_GM12878_OCR_ext2_OCRModel_v20230524_v1.loop", sep='\t')
num_pos = data[data['response'] == 1].shape[0]
num_neg = data[data['response'] == 0].shape[0]
val_chr_list = ["chr22", "chrX"]
train_data = data[data['chrom'].isin(val_chr_list) == False]
val_data = data[data['chrom'].isin(val_chr_list) == True]
train_X = train_data.iloc[:,[4,5,6,7,8,9,15,16,17,18,19,20,21,22]]
val_X = val_data.iloc[:,[4,5,6,7,8,9,15,16,17,18,19,20,21,22]]
# transform dim5 to one-hot
# one_hot = pd.get_dummies(train_X['motif_pattern'], prefix='motif_pattern')
# train_X = pd.concat([one_hot, train_X.drop('motif_pattern', axis=1)], axis=1)
# one_hot = pd.get_dummies(val_X['motif_pattern'], prefix='motif_pattern')
# val_X = pd.concat([one_hot, val_X.drop('motif_pattern', axis=1)], axis=1)
train_X = train_X.to_numpy()
val_X = val_X.to_numpy()
# labels
train_Y = train_data['response'].to_numpy()
val_Y = val_data['response'].to_numpy()
print(f"train data num: {train_Y.shape}")
print(f"val data num: {val_Y.shape}")

class MLLoopDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = MLLoopDataset(train_X, train_Y)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
val_dataset = MLLoopDataset(val_X, val_Y)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
print("="*7)
print(f"len(train_dataloader)==train_batch_num: {len(train_dataloader)}")
print(f"len(val_dataloader)==val_batch_num: {len(val_dataloader)}")


# * model contructor
if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.backends.cudnn.enabled = True # type: ignore
    # torch.backends.cudnn.benchmark = True # type: ignore
else:
    device = torch.device('cpu')
log_info = f"""device: {device}"""
print(log_info)
log_out.write(log_info+"\n")
model = ML_Loop_Model_v20230531()
model.to(device)


# * scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
log_out.write(f"optimizer: {type(optimizer)}\n")
early_stopping = EarlyStopping(mode="min", patience=PATIENCE, verbose=True, delta=DELTA, start_epoch=START_EPOCH, log_file=log_out)
# lambdaLR = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.01*((step+1)/13))
# lambdaLR2 = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
cosineAnnealingLR_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=11, eta_min=LEARNING_RATE/100, last_epoch=-1)
# reduceLROnPlateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=13, threshold=DELTA, min_lr=1E-7, eps=1E-7, verbose=True)
# learningRate_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[lambdaLR, lambdaLR2, cosineAnnealingLR_scheduler], milestones=[2, 60])

# best metrics record
best_val_loss = 1e10
best_val_loss_log_info = None

def compute_loop_loss(pred_scores, mid_outputs, labels, ratio=0.5):
    loss_a = nn.BCELoss()(pred_scores, labels)
    loss_mid = 0
    for i in range(mid_outputs.shape[1]):
        loss_mid += nn.BCELoss()(mid_outputs[:,i], labels)
    # loss_mid /= mid_outputs.shape[1]
    # return ratio*loss_a+(1-ratio)*loss_mid, loss_a
    return (loss_a+loss_mid)/(mid_outputs.shape[1]+1), loss_a

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
        pred_scores, mid_outputs = model(batch_features)
        # 计算损失函数
        loss, loss_final = compute_loop_loss(pred_scores.squeeze(1), mid_outputs, batch_labels, ratio=0.5)
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        if batch_num % LOG_STEP == 0:
            model.eval()
            with torch.no_grad():
                acc = compute_loop_acc(pred_scores, batch_labels)
                val_loss, val_loss_final, val_acc = 0, 0, 0
                val_batch_num = len(val_dataloader)
                for val_batch in val_dataloader:
                    val_batch_features, val_batch_labels = val_batch
                    val_batch_features, val_batch_labels = val_batch_features.to(device), val_batch_labels.to(device)
                    val_pred_scores, val_mid_outputs = model(val_batch_features)
                    temp_val_loss, temp_loss_final = compute_loop_loss(val_pred_scores.squeeze(1), val_mid_outputs, val_batch_labels, ratio=0.5)
                    val_loss += temp_val_loss.item()
                    val_loss_final += temp_loss_final.item()
                    val_acc += compute_loop_acc(val_pred_scores, val_batch_labels)
                val_loss, val_acc, val_loss_final = val_loss/val_batch_num, val_acc/val_batch_num, val_loss_final/val_batch_num
                log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, loss_final: {loss_final.item():.4f}, acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_loss_final: {val_loss_final:.4f}, val_acc: {val_acc:.4f}"
                print(log_info)
                log_out.write(log_info + '\n')
                global_step = e*len(train_dataloader)//(INIT_BATCH_SIZE/BATCH_SIZE)+batch_num//(INIT_BATCH_SIZE/BATCH_SIZE)
                train_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=global_step)
                train_writer.add_scalar(tag='acc', scalar_value=acc, global_step=global_step)
                val_writer.add_scalar(tag='loss', scalar_value=val_loss, global_step=global_step)
                val_writer.add_scalar(tag='acc', scalar_value=val_acc, global_step=global_step)
                if val_loss_final < best_val_loss: # update best val loss
                    best_val_loss = val_loss_final
                    best_val_loss_log_info = f"Epoch {e}\t"+log_info
                    checkpoint = {
                        'epoch':e,
                        'model':model.state_dict()
                    }
                    torch.save(checkpoint, LOOP_MODEL_PATH+f'/model/LoopModel_{log_time}.pt')
            # 早停止
            early_stopping(metric=val_loss_final, current_epoch=e)
            cosineAnnealingLR_scheduler.step()
            # if e < 20:
            #     learningRate_scheduler.step()
            # else:
            #     reduceLROnPlateau_scheduler.step(val_loss)
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
print(f"Best val_loss_final record: {best_val_loss_log_info}")
log_out.write(f"Best val_loss_final record: {best_val_loss_log_info}\n")

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()

