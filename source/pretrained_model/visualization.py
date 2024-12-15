
import torch
import torch.nn as nn
from models import *

# model = PretrainedModel_v20230417_v1()
checkpoint = torch.load(r'./model/PretrainedModel_2023_04_22_02_19_04_GM12878_PretrainedModel_v20230421_v3.pt')
saved_dict = checkpoint['model']
# print(saved_dict.keys())
key_words = ['layer_scale_1', 'layer_scale_2', 'diag_scale_H', 'diag_scale_W']
for k in saved_dict:
    for kw in key_words:
        if kw in k:
            print(k, saved_dict[k])
