
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import itertools
from parameters import LOOP_MODEL_PATH
from dl_loop_models import SelectItem, append_branch
from sklearn.metrics import accuracy_score



def seed_torch(seed=9523):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU. torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False # type: ignore 
    torch.backends.cudnn.deterministic = True  # 选择确定性算法 # type: ignore


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # torch.backends.cudnn.enabled = True # type: ignore
        # torch.backends.cudnn.benchmark = True # type: ignore
    else:
        device = torch.device('cpu')
    return device



def get_loop_model(device):
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
    one_hot = pd.get_dummies(train_X['motif_pattern'], prefix='motif_pattern')
    train_X = pd.concat([one_hot, train_X.drop('motif_pattern', axis=1)], axis=1)
    one_hot = pd.get_dummies(val_X['motif_pattern'], prefix='motif_pattern')
    val_X = pd.concat([one_hot, val_X.drop('motif_pattern', axis=1)], axis=1)
    train_X = train_X.to_numpy()
    val_X = val_X.to_numpy()
    # labels
    train_Y = train_data['response'].to_numpy()
    val_Y = val_data['response'].to_numpy()
    print(f"train data num: {train_Y.shape}")
    print(f"val data num: {val_Y.shape}")
    # * model
    checkpoint = torch.load(LOOP_MODEL_PATH+r'/model/LoopModel_2023_06_23_16_41_10.pt')
    saved_dict = checkpoint['model']
    print(type(saved_dict))
    print(len(saved_dict))
    '''
    每个分支有31层，ensemble_module_list共199个分支。所有分支一共有6169层。
    out_layer有4层。
    '''
    FEATURE_INDEX = [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17]]
    feature_combination = []
    feature_combination = list(itertools.combinations(FEATURE_INDEX, 3))
    temp_comb = list()
    for comb1 in feature_combination:
        temp = list()
        for single in comb1:
            temp += single
        temp_comb.append(temp)
    feature_combination = temp_comb
    feature_combination = [i for i in feature_combination if set(i).intersection(set([0,1,2,3,4,5,6]))]
    for i, f_index in enumerate(feature_combination):
        model = append_branch(depth=4, feature_num=len(f_index))
        new_dict = model.state_dict()
        for key in new_dict:
            new_dict[key] = saved_dict[f"ensemble_module_list.{i}."+key]
        model.load_state_dict(new_dict)
        model.to(device)
        pred_score = model(torch.from_numpy(val_X[:,f_index]).float().to(device))
        pred_Y = list(map(int, pred_score >= 0.5))
        acc = accuracy_score(val_Y, pred_Y)
        print(acc)
        # print(pred_score)
    # i = 0
    # for layer in saved_dict:
    #     if "ensemble_module_list" in layer: continue
    #     i += 1
    #     print(layer)
    # print(i)


if __name__ == '__main__':
    seed_torch()
    device = get_device()
    get_loop_model(device)

