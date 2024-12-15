from train import DeepLncCTCFModel, load_data, load_data_by_health
import torch, random, os, math
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

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
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = DeepLncCTCFModel()
checkpoint = torch.load(r'./model/LoopModel_2023_08_10_17_08_15_8cellbychr.pt')
saved_dict = checkpoint['model']
model.load_state_dict(saved_dict)
model.to(device)
model.eval()


cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
val_cell_line_list = ["Caco-2", "HCT116", "K562", "MCF-7"]
VAL_CHR_LIST = ['chr22', 'chrX']
# train_features,  train_labels, val_features, val_labels = load_data(cell_line_list=cell_line_list, val_chr_list=VAL_CHR_LIST)
train_features,  train_labels, val_features, val_labels = load_data_by_health(cell_line_list=cell_line_list, val_cell_line_list=val_cell_line_list)
print("="*7)
log_info = f"""len(train_features): {train_features.shape}
len(train_labels): {train_labels.shape}
len(val_features): {val_features.shape}
len(val_labels): {val_labels.shape}"""
print(log_info)

BATCH_SIZE = 256
val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
val_labels = torch.tensor(val_labels, dtype=torch.float32).cpu()
test_pred_scores = model(val_features[:BATCH_SIZE].cuda()).cpu()
with torch.no_grad():
    for i in range(1, math.ceil(len(val_labels)/BATCH_SIZE)):
        print(i)
        vstemp = val_features[i*BATCH_SIZE:(i+1)*BATCH_SIZE].cuda()
        print("1", vstemp.shape)
        temp = model(vstemp)
        temp = temp.cpu()
        test_pred_scores = torch.cat([test_pred_scores, temp], dim=0).cpu()
        print(test_pred_scores.shape)
    # test_pred_scores = model(val_features)


def compute_loop_acc(pred1, label1):
    if len(pred1.shape) == 2:
        pred1 = pred1.squeeze(1)
    # 将预测分数四舍五入为0或1，表示预测标签
    pred1 = torch.round(pred1)
    print(pred1.shape)
    print(label1.shape)
    # 计算预测标签与真实标签相同的数量
    correct = (pred1 == label1).sum().item()
    # print(correct, len(label1))
    # 计算精度值
    accuracy = correct / len(label1)
    return accuracy

def compute_roc_auc(predictions, labels):
    # 计算 ROC 曲线下面积（AUC）
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    roc_auc = roc_auc_score(labels, predictions)
    # print("ROC AUC: {:.4f}".format(roc_auc))
    return roc_auc

def compute_prc_auc(predictions, labels):
    # 计算 PRC 曲线下面积（AUC）
    # 其中 labels 是真实的二进制标签，predictions 是模型的预测概率值。
    # 函数会返回 PRC AUC 的值。
    # 注意，average_precision_score 函数要求输入的 predictions 是概率值，而不是二进制标签。
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    prc_auc = average_precision_score(labels, predictions)
    # print("PRC AUC: {:.4f}".format(prc_auc))
    return prc_auc

print("acc: {:.4f}".format(compute_loop_acc(test_pred_scores, val_labels)))
print("ROC AUC: {:.4f}".format(compute_roc_auc(test_pred_scores, val_labels)))
print("PRC AUC: {:.4f}".format(compute_prc_auc(test_pred_scores, val_labels)))

