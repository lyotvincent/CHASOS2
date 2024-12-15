'''
@author: 孙嘉良
'''

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

import random
import numpy as np
from parameters import PRETRAINED_MODEL_PATH
from pretrained_data_loader import load_kmer_encoded_data, PretrainedDataset, load_kmer_encoded_data_div_cell_line
from models import *
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from metrics import compute_two_loss, compute_accuracy, compute_one_loss, compute_one_acc
from scheduler import EarlyStopping

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
log_out = open(PRETRAINED_MODEL_PATH+r'/logs/log_'+log_time+'.txt', 'w', buffering=8)
train_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/train')
val_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val')

# * hyper parameters
SPLIT_MODE = 'chr'
VAL_CHR_LIST = ['chr22', 'chrX']
STRIDE = 1
INIT_BATCH_SIZE = 128 # 使用全部cell line 做chr val的
INIT_BATCH_SIZE = 16 # 使用GM12878 做chr val的
BATCH_SIZE = 32
LOG_STEP = int(100*(INIT_BATCH_SIZE/BATCH_SIZE))
LOG_STEP = 500
EPOCH = 100
LEARNING_RATE = 0.0001
PATIENCE = 40
DELTA = 5E-5
START_EPOCH = 5
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
# cell_line_list = ["GM12878", "HCT116", "IMR-90", "K562"]
cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
# kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list, monomer_train_data_list, monomer_val_data_list = load_kmer_encoded_data(SPLIT_MODE, STRIDE)
# kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list = load_kmer_encoded_data(cell_line_list=cell_line_list, split_mode=SPLIT_MODE, val_chr_list=VAL_CHR_LIST, stride=STRIDE)
val_cell_line_list = ["Caco-2", "HCT116", "K562", "MCF-7"]
kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list = load_kmer_encoded_data_div_cell_line(cell_line_list=cell_line_list, val_cell_line_list=val_cell_line_list)
print("="*7)
log_info = f"""len(kmer_train_data_list): {kmer_train_data_list.shape}\nlen(train_label_list): {train_label_list.shape}\nlen(kmer_val_data_list): {kmer_val_data_list.shape}\nlen(val_label_list): {val_label_list.shape}"""
print(log_info)
log_out.write(log_info+"\n")

# TODO leave one cell line out (LOCLO)
# TODO

train_dataset = PretrainedDataset(kmer_train_data_list, None, train_label_list)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
val_dataset = PretrainedDataset(kmer_val_data_list, None, val_label_list)
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
model = AnchorModel_v20230516_v1()
model.to(device)

# write model graph
train_writer.add_graph(model, (torch.randint(4**5, (2, 996)).to(device)))
val_writer.add_graph(model, (torch.randint(4**5, (2, 996)).to(device)))
# write model info
log_out.write(str(model)+'\n')
# write model parameter number
log_info = f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
log_out.write(log_info+"\n")
print(log_info)

# * scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
log_out.write(f"optimizer: {type(optimizer)}\n")
early_stopping = EarlyStopping(mode="min", patience=PATIENCE, verbose=True, delta=DELTA, start_epoch=START_EPOCH, log_file=log_out)
lambdaLR = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.01*((step+1)/13))
lambdaLR2 = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
cosineAnnealingLR_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=13, eta_min=1e-5, last_epoch=-1)
reduceLROnPlateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=13, threshold=DELTA, min_lr=1E-7, eps=1E-7, verbose=True)
learningRate_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[lambdaLR, lambdaLR2, cosineAnnealingLR_scheduler], milestones=[13, 78])

# best metrics record
best_val_loss = 1e10
best_val_loss_log_info = None
best_val_anchor_acc = 0
best_val_anchor_acc_log_info = None
best_val_ocr_acc = 0
best_val_ocr_acc_log_info = None

# * model train
for e in range(EPOCH):
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
        batch_labels = batch_labels[:, 0] # 使用anchor label，放弃ocr label
        # 梯度清零
        optimizer.zero_grad()
        # 计算模型的输出
        pred_scores = model(batch_kmer_data)
        # 计算损失函数
        # loss = compute_two_loss(pred_scores, batch_labels, ratio=1.)
        loss = compute_one_loss(pred_scores, batch_labels)
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        if batch_num % LOG_STEP == 0:
            model.eval()
            with torch.no_grad():
                # anchor_acc, ocr_acc = compute_accuracy(pred_scores, batch_labels)
                anchor_acc = compute_one_acc(pred_scores, batch_labels)
                val_loss, val_anchor_acc, val_ocr_acc = 0, 0, 0
                val_batch_num = len(val_dataloader)
                for val_batch in val_dataloader:
                    # print(batch_num, 9)
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_mono_data.to(device), val_batch_labels.to(device)
                    val_batch_kmer_data, val_batch_labels = val_batch
                    val_batch_labels = val_batch_labels[:, 0]
                    val_batch_kmer_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_labels.to(device)
                    val_pred_scores = model(val_batch_kmer_data)
                    # val_loss += compute_two_loss(val_pred_scores, val_batch_labels, ratio=1.).item()
                    val_loss += compute_one_loss(val_pred_scores, val_batch_labels).item()
                    # temp_val_anchor_acc, temp_val_ocr_acc = compute_accuracy(val_pred_scores, val_batch_labels)
                    temp_val_anchor_acc = compute_one_acc(val_pred_scores, val_batch_labels)
                    val_anchor_acc += temp_val_anchor_acc
                    # val_ocr_acc += temp_val_ocr_acc
                val_loss, val_anchor_acc, val_ocr_acc = val_loss/val_batch_num, val_anchor_acc/val_batch_num, val_ocr_acc/val_batch_num
                log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, anchor_acc: {anchor_acc:.4f}, ocr_acc: {0:.4f}, val_loss: {val_loss:.4f}, val_anchor_acc: {val_anchor_acc:.4f}, val_ocr_acc: {val_ocr_acc:.4f}"
                print(log_info)
                log_out.write(log_info + '\n')
                global_step = e*len(train_dataloader)//(INIT_BATCH_SIZE/BATCH_SIZE)+batch_num//(INIT_BATCH_SIZE/BATCH_SIZE)
                train_writer.add_scalar(tag='loss', scalar_value=loss, global_step=global_step)
                train_writer.add_scalar(tag='anchor_acc', scalar_value=anchor_acc, global_step=global_step)
                # train_writer.add_scalar(tag='ocr_acc', scalar_value=ocr_acc, global_step=global_step)
                val_writer.add_scalar(tag='loss', scalar_value=val_loss, global_step=global_step)
                val_writer.add_scalar(tag='anchor_acc', scalar_value=val_anchor_acc, global_step=global_step)
                # val_writer.add_scalar(tag='ocr_acc', scalar_value=val_ocr_acc, global_step=global_step)
                # val_writer.add_pr_curve(tag='pr_curve', labels=labels, predictions=predictions, global_step=0)
                if val_loss < best_val_loss: # update best val loss
                    best_val_loss = val_loss
                    best_val_loss_log_info = f"Epoch {e}\t"+log_info
                if (val_anchor_acc > best_val_anchor_acc) or (val_anchor_acc == best_val_anchor_acc and val_ocr_acc > best_val_ocr_acc): # update best val anchor acc
                    best_val_anchor_acc = val_anchor_acc
                    best_val_anchor_acc_log_info = f"Epoch {e}\t"+log_info
                    checkpoint = {
                        'epoch':e,
                        'model':model.state_dict()
                    }
                    torch.save(checkpoint,PRETRAINED_MODEL_PATH+f'/model/PretrainedModel_{log_time}.pt')
                if (val_ocr_acc > best_val_ocr_acc) or (val_ocr_acc == best_val_ocr_acc and val_anchor_acc > best_val_anchor_acc):
                    best_val_ocr_acc = val_ocr_acc
                    best_val_ocr_acc_log_info = f"Epoch {e}\t"+log_info
            # 早停止
            early_stopping(metric=val_loss, current_epoch=e)
            if e < 20:
                learningRate_scheduler.step()
            else:
                reduceLROnPlateau_scheduler.step(val_loss)
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
# print(f"Best val ocr acc record: {best_val_ocr_acc_log_info}")
# log_out.write(f"Best val ocr acc record: {best_val_ocr_acc_log_info}\n")

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()

