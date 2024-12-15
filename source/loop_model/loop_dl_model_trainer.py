'''
@author: 孙嘉良
'''

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained_model'))

import random
import numpy as np
from parameters import LOOP_MODEL_PATH
from preprocess_data import load_loop_dl_data, LoopDataset, load_loop_dl_data_by_cell_line
from dl_loop_models import *
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from loop_dl_metrics import compute_loop_loss, compute_loop_acc, compute_multi_loop_loss, compute_roc_auc, compute_prc_auc
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
LOG_STEP = 50
EPOCH = 80
LEARNING_RATE = 0.01
PATIENCE = 40
DELTA = 5E-5
START_EPOCH = 3
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
# cell_line_list = ["GM12878"]
# cell_line_list = ["GM12878", "HCT116", "IMR-90", "K562"]
cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
log_out.write(f"cell_line_list: {cell_line_list}\n")
# kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list, monomer_train_data_list, monomer_val_data_list = load_kmer_encoded_data(SPLIT_MODE, STRIDE)
# train_features, train_kmers1, train_kmers2, train_labels, val_features, val_kmers1, val_kmers2, val_labels = load_loop_dl_data(cell_line_list=cell_line_list, val_chr_list=VAL_CHR_LIST)
val_cell_line_list = ["Caco-2", "HCT116", "K562", "MCF-7"]
log_out.write(f"val_cell_line_list: {val_cell_line_list}\n")
train_features, train_kmers1, train_kmers2, train_labels, val_features, val_kmers1, val_kmers2, val_labels = load_loop_dl_data_by_cell_line(cell_line_list=cell_line_list, val_cell_line_list=val_cell_line_list)
print("="*7)
log_info = f"""len(train_features): {train_features.shape}
len(train_kmers1): {train_kmers1.shape}
len(train_kmers2): {train_kmers2.shape}
len(train_labels): {train_labels.shape}
len(val_features): {val_features.shape}
len(val_kmers1): {val_kmers1.shape}
len(val_kmers2): {val_kmers2.shape}
len(val_labels): {val_labels.shape}"""
print(log_info)
log_out.write(log_info+"\n")

train_dataset = LoopDataset(train_features, train_kmers1, train_kmers2, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
val_dataset = LoopDataset(val_features, val_kmers1, val_kmers2, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
print("="*7)
log_info = f"len(train_dataloader)==train_batch_num: {len(train_dataloader)}"
print(log_info)
log_out.write(log_info+"\n")
log_info = f"len(val_dataloader)==val_batch_num: {len(val_dataloader)}"
print(log_info)
log_out.write(log_info+"\n")


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
model = DL_Loop_Model_v20230621()
# checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_03_30_22_52_07_GM12878_PretrainedModel_v20230330_vAddALayer.pt')
# saved_dict = checkpoint['model']
# saved_dict['out_layer.4.weight'] = saved_dict['out_layer.4.weight'][1:2, :]
# saved_dict['out_layer.4.bias'] = saved_dict['out_layer.4.bias'][1:2]
# model.load_state_dict(saved_dict)
model.to(device)


# * scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
log_out.write(f"optimizer: {type(optimizer)}\n")
early_stopping = EarlyStopping(mode="min", patience=PATIENCE, verbose=True, delta=DELTA, start_epoch=START_EPOCH, log_file=log_out)
lambdaLR = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.01*((step+1)/13))
lambdaLR2 = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
cosineAnnealingLR_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=LEARNING_RATE/32, last_epoch=-1) # type: ignore
reduceLROnPlateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=13, threshold=DELTA, min_lr=1E-7, eps=1E-7, verbose=True)
learningRate_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[lambdaLR, lambdaLR2, cosineAnnealingLR_scheduler], milestones=[2, 60])

# best metrics record
best_val_loss = 1e10
best_val_loss_log_info = None
best_val_acc = 0
best_val_acc_log_info = None
best_val_roc_auc = 0
best_val_roc_auc_log_info = None
best_val_prc_auc = 0
best_val_prc_auc_log_info = None

# * model train
val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
for e in range(EPOCH):
    print(f"Epoch {e}")
    log_out.write(f"Epoch {e}\n")
    print('----------')
    log_out.write('----------\n')

    for batch_num, batch in enumerate(train_dataloader):
        model.train()
        batch_features, batch_kmer_data_1, batch_kmer_data_2, batch_labels = batch
        batch_features, batch_kmer_data_1, batch_kmer_data_2, batch_labels = batch_features.to(device), batch_kmer_data_1.to(device), batch_kmer_data_2.to(device), batch_labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 计算模型的输出
        pred_scores, mid_outputs = model(batch_features, batch_kmer_data_1, batch_kmer_data_2)
        # 计算损失函数
        # loss = compute_loop_loss(pred_scores, batch_labels.unsqueeze(1))
        loss, loss_final = compute_multi_loop_loss(pred_scores.squeeze(1), mid_outputs, batch_labels)
        roc_auc = compute_roc_auc(pred_scores, batch_labels)
        prc_auc = compute_prc_auc(pred_scores, batch_labels)
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        if batch_num % LOG_STEP == 0:
            model.eval()
            with torch.no_grad():
                acc = compute_loop_acc(pred_scores, batch_labels)
                # val_loss, val_loss_final, val_acc = 0, 0, 0
                val_pred_scores, val_mid_outputs = torch.Tensor(), torch.Tensor()
                val_batch_num = len(val_dataloader)
                for val_batch in val_dataloader:
                    val_batch_features, val_batch_kmer_data_1, val_batch_kmer_data_2, val_batch_labels = val_batch
                    val_batch_features, val_batch_kmer_data_1, val_batch_kmer_data_2, val_batch_labels = val_batch_features.to(device), val_batch_kmer_data_1.to(device), val_batch_kmer_data_2.to(device), val_batch_labels.to(device)
                    temp_val_pred_scores, temp_val_mid_outputs = model(val_batch_features, val_batch_kmer_data_1, val_batch_kmer_data_2)
                    if len(val_pred_scores) == 0:
                        val_pred_scores = temp_val_pred_scores
                        val_mid_outputs = temp_val_mid_outputs
                    else:
                        val_pred_scores = torch.cat((val_pred_scores, temp_val_pred_scores), dim=0)
                        val_mid_outputs = torch.cat((val_mid_outputs, temp_val_mid_outputs), dim=0)
                #     temp_val_loss, temp_loss_final = compute_multi_loop_loss(val_pred_scores.squeeze(1), val_mid_outputs, val_batch_labels, ratio=0.5)
                #     val_loss += temp_val_loss.item()
                #     val_loss_final += temp_loss_final.item()
                #     val_acc += compute_loop_acc(val_pred_scores, val_batch_labels)
                # val_loss, val_acc, val_loss_final = val_loss/val_batch_num, val_acc/val_batch_num, val_loss_final/val_batch_num
                val_loss, val_loss_final = compute_multi_loop_loss(val_pred_scores.squeeze(1), val_mid_outputs, val_labels)
                val_acc = compute_loop_acc(val_pred_scores, val_labels)
                val_roc_auc = compute_roc_auc(val_pred_scores, val_labels)
                val_prc_auc = compute_prc_auc(val_pred_scores, val_labels)
                log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, loss_final: {loss_final.item():.4f}, acc: {acc:.4f}, roc_auc: {roc_auc:.4f}, prc_auc: {prc_auc:.4f}, val_loss: {val_loss:.4f}, val_loss_final: {val_loss_final:.4f}, val_acc: {val_acc:.4f}, val_roc_auc: {val_roc_auc:.4f}, val_prc_auc: {val_prc_auc:.4f}"
                print(log_info)
                log_out.write(log_info + '\n')
                global_step = e*len(train_dataloader)//(INIT_BATCH_SIZE/BATCH_SIZE)+batch_num//(INIT_BATCH_SIZE/BATCH_SIZE)
                train_writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=global_step)
                train_writer.add_scalar(tag='acc', scalar_value=acc, global_step=global_step)
                val_writer.add_scalar(tag='loss', scalar_value=val_loss, global_step=global_step)
                val_writer.add_scalar(tag='acc', scalar_value=val_acc, global_step=global_step)
                if e > 1 and val_loss_final < best_val_loss: # update best val loss
                    best_val_loss = val_loss_final
                    best_val_loss_log_info = f"Epoch {e}\t"+log_info
                if e > 1 and val_acc > best_val_acc: # update best val acc
                    best_val_acc = val_acc
                    best_val_acc_log_info = f"Epoch {e}\t"+log_info
                if e > 1 and val_roc_auc > best_val_roc_auc:
                    best_val_roc_auc = val_roc_auc
                    best_val_roc_auc_log_info = f"Epoch {e}\t"+log_info
                    checkpoint = {
                        'epoch':e,
                        'model':model.state_dict()
                    }
                    torch.save(checkpoint, LOOP_MODEL_PATH+f'/model/LoopModel_{log_time}.pt')
                if e > 2 and val_prc_auc > best_val_prc_auc:
                    best_val_prc_auc = val_prc_auc
                    best_val_prc_auc_log_info = f"Epoch {e}\t"+log_info
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
print(f"Best val_acc record: {best_val_acc_log_info}")
log_out.write(f"Best val_acc record: {best_val_acc_log_info}\n")
print(f"Best val_roc_auc record: {best_val_roc_auc_log_info}")
log_out.write(f"Best val_roc_auc record: {best_val_roc_auc_log_info}\n")
print(f"Best val_prc_auc record: {best_val_prc_auc_log_info}")
log_out.write(f"Best val_prc_auc record: {best_val_prc_auc_log_info}\n")

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()


