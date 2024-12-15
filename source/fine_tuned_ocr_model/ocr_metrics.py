
import torch.nn as nn
import torch

def compute_ocr_loss(pred_scores, labels):
    loss_a = nn.MSELoss()(pred_scores, labels)
    return loss_a

def compute_ocr_1000_loss(pred_scores, labels, ratio=0.5):
    '''
    @input: pred_scores: (batch_size, 1000)
            labels: (batch_size, 1000)
    '''
    loss_1 = nn.MSELoss()(pred_scores, labels)
    loss_avg = nn.MSELoss()(torch.mean(pred_scores, dim=1), torch.mean(labels, dim=1))
    return ratio * loss_1 + (1-ratio) * loss_avg

def compute_ocr_rc_loss(pred_scores, labels, ratio=0.5, threshold=0.3):
    '''
    @description: compute the loss of both regression and classification
    @input: pred_scores: (batch_size, 1000)
            labels: (batch_size, 1000)
    '''
    loss_r = nn.MSELoss()(pred_scores, labels)
    classification_labels = torch.where(labels > threshold, torch.ones_like(labels), torch.zeros_like(labels))  # 根据比例将元素设为1.0或0.0
    loss_c = nn.MSELoss()(pred_scores, classification_labels)
    return ratio * loss_r + (1-ratio) * loss_c

