
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from parameters import PRETRAINED_MODEL_PATH
from models import PretrainedLayers

model = PretrainedLayers()
checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/pretrained_layer/model/PretrainedLayers_best.pth')
start_epoch = checkpoint['epoch']
saved_dict = checkpoint['net']
model.load_state_dict(checkpoint['net'])
print('load model from epoch: ', start_epoch)
for var_name in saved_dict:
    print(var_name, "\t", saved_dict[var_name].size())
# print({saved_dict[var_name].size():var_name for var_name in saved_dict})


# a = torch.randn(1, 16, 5, 2)

# conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), padding='same', dilation=2)

# b = conv1(a)
# print(b)

# weight_dict = {'weight': saved_dict['conv_3.2.weight']}
# conv1_dict = conv1.state_dict()
# conv1_dict.update(weight_dict)
# conv1.load_state_dict(conv1_dict)

# c = conv1(a)
# print(c)
            

'''
load model from epoch:  44
embedding_1.weight       torch.Size([1024, 128])
pos_encoding.pe          torch.Size([1, 1000, 128])
conv_3.0.weight          torch.Size([16, 1, 1, 1])
conv_3.0.bias    torch.Size([16])
conv_3.1.weight          torch.Size([16, 16, 1, 3])
conv_3.1.bias    torch.Size([16])
conv_3.2.weight          torch.Size([16, 16, 3, 1])
conv_3.2.bias    torch.Size([16])
conv_3.4.weight          torch.Size([32, 16, 1, 1])
conv_3.4.bias    torch.Size([32])
conv_3.5.weight          torch.Size([32, 32, 1, 3])
conv_3.5.bias    torch.Size([32])
conv_3.6.weight          torch.Size([32, 32, 3, 1])
conv_3.6.bias    torch.Size([32])
conv_3.8.weight          torch.Size([64, 32, 1, 1])
conv_3.8.bias    torch.Size([64])
conv_3.9.weight          torch.Size([64, 64, 1, 3])
conv_3.9.bias    torch.Size([64])
conv_3.10.weight         torch.Size([64, 64, 3, 1])
conv_3.10.bias   torch.Size([64])
conv_3.12.weight         torch.Size([128, 64, 1, 1])
conv_3.12.bias   torch.Size([128])
conv_3.13.weight         torch.Size([128, 128, 1, 3])
conv_3.13.bias   torch.Size([128])
conv_3.14.weight         torch.Size([128, 128, 3, 1])
conv_3.14.bias   torch.Size([128])
conv_5.0.weight          torch.Size([16, 1, 1, 1])
conv_5.0.bias    torch.Size([16])
conv_5.1.weight          torch.Size([16, 16, 1, 5])
conv_5.1.bias    torch.Size([16])
conv_5.2.weight          torch.Size([16, 16, 5, 1])
conv_5.2.bias    torch.Size([16])
conv_5.4.weight          torch.Size([32, 16, 1, 1])
conv_5.4.bias    torch.Size([32])
conv_5.5.weight          torch.Size([32, 32, 1, 5])
conv_5.5.bias    torch.Size([32])
conv_5.6.weight          torch.Size([32, 32, 5, 1])
conv_5.6.bias    torch.Size([32])
conv_5.8.weight          torch.Size([64, 32, 1, 1])
conv_5.8.bias    torch.Size([64])
conv_5.9.weight          torch.Size([64, 64, 1, 5])
conv_5.9.bias    torch.Size([64])
conv_5.10.weight         torch.Size([64, 64, 5, 1])
conv_5.10.bias   torch.Size([64])
conv_5.12.weight         torch.Size([128, 64, 1, 1])
conv_5.12.bias   torch.Size([128])
conv_5.13.weight         torch.Size([128, 128, 1, 5])
conv_5.13.bias   torch.Size([128])
conv_5.14.weight         torch.Size([128, 128, 5, 1])
conv_5.14.bias   torch.Size([128])
out_layer.2.weight       torch.Size([64, 256])
out_layer.2.bias         torch.Size([64])
out_layer.4.weight       torch.Size([2, 64])
out_layer.4.bias         torch.Size([2])
'''