'''
@author: 孙嘉良
'''

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import PRETRAINED_MODEL_PATH
from pretrained_data_loader import load_kmer_encoded_data, PretrainedDataset
from models import *
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from metrics import compute_two_loss, compute_accuracy, compute_cosine_similarity_2
from scheduler import EarlyStopping


# * prepare log
start_time = time.time()
log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
log_out = open(PRETRAINED_MODEL_PATH+r'/logs/log_'+log_time+'.txt', 'w', buffering=8)

loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/loss')
sl_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/sl_loss')
sd_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/sd_loss')
cos_sim_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/cos_sim_loss')
val_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val_loss')
val_sl_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val_sl_loss')
val_sd_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val_sd_loss')
val_cos_sim_loss_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/val_cos_sim_loss')
saol_train_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/saol_train_acc')
gap_fc_train_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/gap_fc_train_acc')
saol_val_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/saol_val_acc')
gap_fc_val_writer = SummaryWriter(PRETRAINED_MODEL_PATH+f'/logs/tensorboard_log_{log_time}/gap_fc_val_acc')

# * hyper parameters
SPLIT_MODE = 'chr'
VAL_CHR_LIST = ['chr22', 'chrX']
STRIDE = 1
INIT_BATCH_SIZE = 128 # 使用全部cell line 做chr val的
INIT_BATCH_SIZE = 16 # 使用GM12878 做chr val的
BATCH_SIZE = 16
LOG_STEP = 100*(INIT_BATCH_SIZE//BATCH_SIZE)
EPOCH = 100
LEARNING_RATE = 0.0001
PATIENCE = 150
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
# kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list, monomer_train_data_list, monomer_val_data_list = load_kmer_encoded_data(SPLIT_MODE, STRIDE)
kmer_train_data_list, train_label_list, kmer_val_data_list, val_label_list = load_kmer_encoded_data(cell_line_list=["GM12878"], split_mode=SPLIT_MODE, val_chr_list=VAL_CHR_LIST, stride=STRIDE)
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
    torch.backends.cudnn.enabled = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
else:
    device = torch.device('cpu')
log_info = f"""device: {device}"""
print(log_info)
log_out.write(log_info+"\n")
model = PretrainedModel_v20230326_vSAOL()
model.to(device)

# write model graph
# train_writer.add_graph(model, (torch.randint(4**5, (2, 996)).to(device)))
# val_writer.add_graph(model, (torch.randint(4**5, (2, 996)).to(device)))
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
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//2, threshold=DELTA, min_lr=1E-7, eps=1E-7, verbose=True)

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
        # 梯度清零
        optimizer.zero_grad()
        # 计算模型的输出
        saol_scores, gap_fc_scores = model(batch_kmer_data)
        # 计算损失函数
        sl_loss = compute_two_loss(saol_scores, batch_labels) # supervised loss
        sd_loss = compute_two_loss(gap_fc_scores, batch_labels) # self-distillation loss
        cos_sim_loss = compute_cosine_similarity_2(saol_scores, gap_fc_scores) # cosine similarity loss
        loss = sl_loss + 0.7 * cos_sim_loss + 0.7 * sd_loss
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        if batch_num % LOG_STEP == 0:
            model.eval()
            with torch.no_grad():
                saol_anchor_acc, saol_ocr_acc = compute_accuracy(saol_scores, batch_labels)
                gap_fc_anchor_acc, gap_fc_ocr_acc = compute_accuracy(gap_fc_scores, batch_labels)
                val_loss, val_sl_loss, val_sd_loss, val_cos_sim_loss, val_saol_anchor_acc, val_saol_ocr_acc, val_gap_fc_anchor_acc, val_gap_fc_ocr_acc = 0, 0, 0, 0, 0, 0, 0, 0
                val_batch_num = len(val_dataloader)
                for val_batch in val_dataloader:
                    # print(batch_num, 9)
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch
                    # val_batch_kmer_data, val_batch_mono_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_mono_data.to(device), val_batch_labels.to(device)
                    val_batch_kmer_data, val_batch_labels = val_batch
                    val_batch_kmer_data, val_batch_labels = val_batch_kmer_data.to(device), val_batch_labels.to(device)
                    val_saol_scores, val_gap_fc_scores = model(val_batch_kmer_data)
                    val_sl_loss += compute_two_loss(val_saol_scores, val_batch_labels).item()
                    val_sd_loss += compute_two_loss(val_gap_fc_scores, val_batch_labels).item()
                    val_cos_sim_loss += compute_cosine_similarity_2(val_saol_scores, val_gap_fc_scores)
                    temp_val_saol_anchor_acc, temp_val_saol_ocr_acc = compute_accuracy(val_saol_scores, val_batch_labels)
                    temp_val_gap_fc_anchor_acc, temp_val_gap_fc_ocr_acc = compute_accuracy(val_gap_fc_scores, val_batch_labels)
                    val_saol_anchor_acc += temp_val_saol_anchor_acc
                    val_saol_ocr_acc += temp_val_saol_ocr_acc
                    val_gap_fc_anchor_acc += temp_val_gap_fc_anchor_acc
                    val_gap_fc_ocr_acc += temp_val_gap_fc_ocr_acc
                val_sl_loss, val_sd_loss, val_cos_sim_loss, val_saol_anchor_acc, val_saol_ocr_acc, val_gap_fc_anchor_acc, val_gap_fc_ocr_acc = val_sl_loss/val_batch_num, val_sd_loss/val_batch_num, val_cos_sim_loss/val_batch_num, val_saol_anchor_acc/val_batch_num, val_saol_ocr_acc/val_batch_num, val_gap_fc_anchor_acc/val_batch_num, val_gap_fc_ocr_acc/val_batch_num
                val_loss = val_sl_loss + 0.7 * val_cos_sim_loss + 0.7 * val_sd_loss
                log_info = f"Batch {batch_num}, loss: {loss.item():.4f}, sl_loss: {sl_loss.item():.4f}, sd_loss: {sd_loss.item():.4f}, cos_sim_loss: {cos_sim_loss.item():.4f}, saol_anchor_acc: {saol_anchor_acc:.4f}, saol_ocr_acc: {saol_ocr_acc:.4f}, gap_fc_anchor_acc: {gap_fc_anchor_acc:.4f}, gap_fc_ocr_acc: {gap_fc_ocr_acc:.4f}, val_loss: {val_loss:.4f}, val_sl_loss: {val_sl_loss:.4f}, val_sd_loss: {val_sd_loss:.4f}, val_cos_sim_loss: {val_cos_sim_loss:.4f}, val_saol_anchor_acc: {val_saol_anchor_acc:.4f}, val_saol_ocr_acc: {val_saol_ocr_acc:.4f}, val_gap_fc_anchor_acc: {val_gap_fc_anchor_acc:.4f}, val_gap_fc_ocr_acc: {val_gap_fc_ocr_acc:.4f}"
                print(log_info)
                log_out.write(log_info + '\n')
                global_step = e*len(train_dataloader)//(INIT_BATCH_SIZE//BATCH_SIZE)+batch_num//(INIT_BATCH_SIZE//BATCH_SIZE)
                loss_writer.add_scalar('loss', loss.item(), global_step)
                sl_loss_writer.add_scalar('loss', sl_loss.item(), global_step)
                sd_loss_writer.add_scalar('loss', sd_loss.item(), global_step)
                cos_sim_loss_writer.add_scalar('loss', cos_sim_loss.item(), global_step)
                val_loss_writer.add_scalar('loss', val_loss, global_step)
                val_sl_loss_writer.add_scalar('loss', val_sl_loss, global_step)
                val_sd_loss_writer.add_scalar('loss', val_sd_loss, global_step)
                val_cos_sim_loss_writer.add_scalar('loss', val_cos_sim_loss, global_step)
                saol_train_writer.add_scalar(tag='anchor_acc', scalar_value=saol_anchor_acc, global_step=global_step)
                saol_train_writer.add_scalar(tag='ocr_acc', scalar_value=saol_ocr_acc, global_step=global_step)
                gap_fc_train_writer.add_scalar(tag='anchor_acc', scalar_value=gap_fc_anchor_acc, global_step=global_step)
                gap_fc_train_writer.add_scalar(tag='ocr_acc', scalar_value=gap_fc_ocr_acc, global_step=global_step)
                saol_val_writer.add_scalar(tag='anchor_acc', scalar_value=val_saol_anchor_acc, global_step=global_step)
                saol_val_writer.add_scalar(tag='ocr_acc', scalar_value=val_saol_ocr_acc, global_step=global_step)
                gap_fc_val_writer.add_scalar(tag='anchor_acc', scalar_value=val_gap_fc_anchor_acc, global_step=global_step)
                gap_fc_val_writer.add_scalar(tag='ocr_acc', scalar_value=val_gap_fc_ocr_acc, global_step=global_step)
                # val_writer.add_pr_curve(tag='pr_curve', labels=labels, predictions=predictions, global_step=0)
                if val_sl_loss < best_val_loss: # update best val loss
                    best_val_loss = val_sl_loss
                    best_val_loss_log_info = f"Epoch {e}\t"+log_info
                if (val_saol_anchor_acc > best_val_anchor_acc) or (val_saol_anchor_acc == best_val_anchor_acc and val_saol_ocr_acc > best_val_ocr_acc): # update best val anchor acc
                    best_val_anchor_acc = val_saol_anchor_acc
                    best_val_anchor_acc_log_info = f"Epoch {e}\t"+log_info
                if (val_saol_ocr_acc > best_val_ocr_acc) or (val_saol_ocr_acc == best_val_ocr_acc and val_saol_anchor_acc > best_val_anchor_acc):
                    best_val_ocr_acc = val_saol_ocr_acc
                    best_val_ocr_acc_log_info = f"Epoch {e}\t"+log_info
            # 早停止
            early_stopping(metric=val_loss, current_epoch=e)
            lr_scheduler.step(val_loss)
        #达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            log_info = f"Early stopping at epoch {e}, batch {batch_num}"
            print(log_info)
            log_out.write(f'{log_info}\n')
            break #跳出迭代，结束训练
    if early_stopping.early_stop:
        break

loss_writer.close()
sl_loss_writer.close()
sd_loss_writer.close()
cos_sim_loss_writer.close()
val_loss_writer.close()
val_sl_loss_writer.close()
val_sd_loss_writer.close()
val_cos_sim_loss_writer.close()
saol_train_writer.close()
gap_fc_train_writer.close()
saol_val_writer.close()
gap_fc_val_writer.close()

print('----------')
log_out.write('----------\n')
print(f"Best val loss record: {best_val_loss_log_info}")
log_out.write(f"Best val loss record: {best_val_loss_log_info}\n")
print(f"Best val anchor acc record: {best_val_anchor_acc_log_info}")
log_out.write(f"Best val anchor acc record: {best_val_anchor_acc_log_info}\n")
print(f"Best val ocr acc record: {best_val_ocr_acc_log_info}")
log_out.write(f"Best val ocr acc record: {best_val_ocr_acc_log_info}\n")

run_time = time.time() - start_time
print(f'Run time: {run_time}s')
log_out.write(f"Run time: {run_time}s\n")
log_out.close()

