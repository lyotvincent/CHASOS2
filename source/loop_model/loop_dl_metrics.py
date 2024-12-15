
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_loop_loss(pred_scores, labels):
    loss_a = nn.BCELoss()(pred_scores, labels)
    return loss_a

def compute_multi_loop_loss(pred_scores, mid_outputs, labels, ratio=0.5):
    loss_a = nn.BCELoss()(pred_scores, labels)
    loss_mid = 0
    for i in range(mid_outputs.shape[1]):
        loss_mid += nn.BCELoss()(mid_outputs[:,i], labels)
    # loss_mid /= mid_outputs.shape[1]
    # return ratio*loss_a+(1-ratio)*loss_mid, loss_a
    return (loss_a+loss_mid)/(mid_outputs.shape[1]+1), loss_a

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


