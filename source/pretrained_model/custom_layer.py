

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ref_block'))
from parameters import PRETRAINED_MODEL_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from ref_block.PRMLayer import PRMLayer



def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = torch.div(num_channels, groups, rounding_mode='floor')

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width) # type: ignore

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BatchNormReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class FireBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: squeeze and 1-path expand, from SqueezeNet
    '''
    def __init__(self, input_channels, squeeze_channels, expand_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand_channels, kernel_size=(3, 3), padding='same', groups=groups)
        self.expand_3_bn_relu = BatchNormReLU(expand_channels)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_3 = self.expand_3(s)
        e_3 = self.expand_3_bn_relu(e_3)
        return e_3

class FireBlock_v2(nn.Module):
    '''
    @date: 2023.03.18
    @description: squeeze and 2-path expand
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1))
        self.expand_1_bn_relu = BatchNormReLU(e_1_channels)
        self.expand_3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(3, 3), padding='same', groups=groups)
        self.expand_3_bn_relu = BatchNormReLU(e_3_channels)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_1 = self.expand_1_bn_relu(e_1)
        e_3 = self.expand_3(s)
        e_3 = self.expand_3_bn_relu(e_3)
        x = torch.cat([e_1, e_3], dim=1)
        return x

class FireBlock_v3(nn.Module):
    '''
    @date: 2023.03.19
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            BatchNormReLU(e_5_channels)
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v4(nn.Module):
    '''
    @date: 2023.03.21
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on v3,
                  remove the BatchNormReLU in the squeeze & expand layers, only add a BatchNormReLU in the output
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
        )
        self.bnr = BatchNormReLU(e_1_channels+e_3_channels+e_5_channels)

    def forward(self, x):
        s = self.squeeze(x)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.bnr(x)
        return x

class FireBlock_v5(nn.Module):
    '''
    @date: 2023.03.26
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on v3,
                  replace BatchNorm with LayerNorm
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, ln_size, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.LayerNorm([squeeze_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.LayerNorm([e_1_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            nn.LayerNorm([e_3_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            nn.LayerNorm([e_5_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )

    def forward(self, x):
        s = self.squeeze(x)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v6(nn.Module):
    '''
    @date: 2023.03.19
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on FireBlock_v3
                  replace kernel 1*3 & dilation 2 with kernel 1*5
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', groups=groups),
            BatchNormReLU(e_5_channels)
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v7(nn.Module):
    '''
    @date: 2023.03.28
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on FireBlock_v3
                  init 1*3 3*1 weight bias with pretraind layer
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            BatchNormReLU(e_5_channels)
        )

        self._init_by_pretrained_layers()

    def _init_by_pretrained_layers(self):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/pretrained_layer/model/PretrainedLayers_best.pth')
        saved_dict = checkpoint['net']
        expand_3_state_dict = self.expand_3.state_dict()
        for param_tensor_name in expand_3_state_dict:
            weight_name, bias_name = get_init_layer_name(expand_3_state_dict[param_tensor_name].size())
            if weight_name:
                expand_3_state_dict[param_tensor_name] = saved_dict[weight_name]
                expand_3_state_dict[param_tensor_name.replace('weight', 'bias')] = saved_dict[bias_name]
        self.expand_3.load_state_dict(expand_3_state_dict)
        expand_5_state_dict = self.expand_5.state_dict()
        for param_tensor_name in expand_5_state_dict:
            weight_name, bias_name = get_init_layer_name(expand_5_state_dict[param_tensor_name].size())
            if weight_name:
                expand_5_state_dict[param_tensor_name] = saved_dict[weight_name]
                expand_5_state_dict[param_tensor_name.replace('weight', 'bias')] = saved_dict[bias_name]
        self.expand_5.load_state_dict(expand_5_state_dict)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x


class LargeKernelFireBlock_v1(nn.Module):
    '''
    @date: 2023.04.12
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  use large kernel
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=3, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=3, groups=groups),
            BatchNormReLU(e_5_channels)
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class LargeKernelFireBlock_v2(nn.Module):
    '''
    @date: 2023.04.12
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  use large kernel
                  replace BN with LN
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze_ln = LayerNorm(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=3, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=3, groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_ln(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v3(nn.Module):
    '''
    @date: 2023.04.12
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  use large kernel
                  replace BN with LN
                  use DW PW
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels):
        super().__init__()
        # squeeze
        self.squeeze_ln = LayerNorm(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same')
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same'),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=e_3_channels),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=e_3_channels),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same'),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=3, groups=e_5_channels),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=3, groups=e_5_channels),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same'),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        x = self.squeeze_ln(x)
        x = self.squeeze(x)
        x = self.squeeze_act(x)
        e_1 = self.expand_1(x)
        e_3 = self.expand_3(x)
        e_5 = self.expand_5(x)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v4(nn.Module):
    '''
    @date: 2023.04.15
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  use large kernel
                  replace LN in LargeKernelFireBlock_v2 with BN
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v5(nn.Module):
    '''
    @date: 2023.04.16
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v6(nn.Module):
    '''
    @date: 2023.04.18
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 5
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_5(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v7(nn.Module):
    '''
    @date: 2023.04.18
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 5
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=(1, 2), groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=(2, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_5(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v8(nn.Module):
    '''
    @date: 2023.04.18
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v9(nn.Module):
    '''
    @date: 2023.04.18
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', dilation=(1, 2), groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', dilation=(2, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v10(nn.Module):
    '''
    @date: 2023.04.19
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 5
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=1, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=1, groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_5(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v11(nn.Module):
    '''
    @date: 2023.04.19
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 7
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_7_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_7 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_7_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_7_channels, out_channels=e_7_channels, kernel_size=(1, 7), padding='same', dilation=1, groups=groups),
            nn.Conv2d(in_channels=e_7_channels, out_channels=e_7_channels, kernel_size=(7, 1), padding='same', dilation=1, groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_7(s)
        x = torch.cat([e_1, e_3], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v12(nn.Module):
    '''
    @date: 2023.04.19
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 31, 33
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_31_channels, e_33_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_31 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_31_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(1, 3), padding='same', dilation=1, groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(3, 1), padding='same', dilation=1, groups=groups),
        )
        self.expand_33 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_33_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_33_channels, out_channels=e_33_channels, kernel_size=(1, 3), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_33_channels, out_channels=e_33_channels, kernel_size=(3, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_31 = self.expand_31(s)
        e_33 = self.expand_33(s)
        x = torch.cat([e_1, e_31, e_33], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v13(nn.Module):
    '''
    @date: 2023.04.19
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 31, 33
    '''
    def __init__(self, input_channels, squeeze_channels, e_31_channels, e_33_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_31 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_31_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(1, 3), padding='same', dilation=1, groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(3, 1), padding='same', dilation=1, groups=groups),
        )
        self.expand_33 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_33_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_33_channels, out_channels=e_33_channels, kernel_size=(1, 3), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_33_channels, out_channels=e_33_channels, kernel_size=(3, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_31 = self.expand_31(s)
        e_33 = self.expand_33(s)
        x = torch.cat([e_31, e_33], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v14(nn.Module):
    '''
    @date: 2023.04.19
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 31, 53
    '''
    def __init__(self, input_channels, squeeze_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', dilation=(1, 3), groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', dilation=(3, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_3, e_5], dim=1)
        x = self.expand_act(x)
        return x

class LargeKernelFireBlock_v15(nn.Module):
    '''
    @date: 2023.05.13
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  kernel 1, 31, 32
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_31_channels, e_32_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(input_channels)
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_31 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_31_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_31_channels, out_channels=e_31_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_32 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_32_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_32_channels, out_channels=e_32_channels, kernel_size=(1, 3), padding='same', dilation=(1, 2), groups=groups),
            nn.Conv2d(in_channels=e_32_channels, out_channels=e_32_channels, kernel_size=(3, 1), padding='same', dilation=(2, 1), groups=groups),
        )
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_31 = self.expand_31(s)
        e_32 = self.expand_32(s)
        x = torch.cat([e_1, e_31, e_32], dim=1)
        x = self.expand_act(x)
        return x

class SEBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: squeeze and excitation
    '''
    def __init__(self, h_channels, reduction):
        super().__init__()
        self.sequential = nn.Sequential(
            # squeeze
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            # excitation
            nn.Linear(h_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU(),
            nn.Linear(reduction, h_channels),
            nn.BatchNorm1d(h_channels),
            nn.Sigmoid()
        )

    def forward(self, h):
        s = self.sequential(h)
        # squeeze & excitation
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        # scale
        x = torch.mul(h, s)
        return x

class SEBlock_v2(nn.Module):
    '''
    @date: 2023.03.21
    @description: squeeze and excitation
                  this block is used for get attention, so remove the ReLU, just use linear
    '''
    def __init__(self, h_channels, reduction):
        super().__init__()
        self.sequential = nn.Sequential(
            # squeeze
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            # excitation
            nn.BatchNorm1d(h_channels),
            nn.Linear(h_channels, reduction),
            nn.Linear(reduction, h_channels),
            nn.Sigmoid()
        )

    def forward(self, h):
        s = self.sequential(h)
        # squeeze & excitation
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        # scale
        x = torch.mul(h, s)
        return x

class SEBlock_v3(nn.Module):
    '''
    @date: 2023.03.15
    @description: squeeze and excitation
    '''
    def __init__(self, h_channels, reduction):
        super().__init__()
        self.sequential = nn.Sequential(
            # squeeze
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            # excitation
            nn.Linear(h_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.GELU(),
            nn.Linear(reduction, h_channels),
            nn.BatchNorm1d(h_channels),
            nn.Sigmoid()
        )

    def forward(self, h):
        s = self.sequential(h)
        # squeeze & excitation
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        # scale
        x = torch.mul(h, s)
        return x

class SKBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: split, fuse and select
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v2(nn.Module):
    '''
    @date: 2023.03.19
    @description: split, fuse and select
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v3(nn.Module):
    '''
    @date: 2023.03.21
    @description: split, fuse and select
                  based on v2,
                  cancel the BatchNormReLU in select part, add the BatchNormReLU after select part
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same')
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same')
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)
        self.bnr = BatchNormReLU(out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        v = self.bnr(v)
        return v

class SKBlock_v4(nn.Module):
    '''
    @date: 2023.03.26
    @description: split, fuse and select
                  based on v2,
                  replace BatchNorm with LayerNorm
    '''
    def __init__(self, h_channels, out_channels, reduction, ln_size):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v5(nn.Module):
    '''
    @date: 2023.03.19
    @description: split, fuse and select
                  based on SKBlock_v2,
                  replace kernel 1*3 & dilation 2 with kernel 1*5
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 5), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(5, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v6(nn.Module):
    '''
    @date: 2023.03.28
    @description: split, fuse and select,
                  based on SKBlock_v2,
                  init 1*3 3*1 weight bias with pretraind layer
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

        self._init_by_pretrained_layers()

    def _init_by_pretrained_layers(self):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/pretrained_layer/model/PretrainedLayers_best.pth')
        saved_dict = checkpoint['net']

        u_3_state_dict = self.u_3.cuda().state_dict()
        for origin_param_name in u_3_state_dict:
            weight_name, bias_name = get_init_layer_name(u_3_state_dict[origin_param_name].size())
            if weight_name:
                u_3_state_dict[origin_param_name] = 0.5*saved_dict[weight_name] + 0.5*u_3_state_dict[origin_param_name]
                u_3_state_dict[origin_param_name.replace('weight', 'bias')] = 0.5*saved_dict[bias_name] + 0.5*u_3_state_dict[origin_param_name.replace('weight', 'bias')]
        self.u_3.load_state_dict(u_3_state_dict)

        u_5_state_dict = self.u_5.cuda().state_dict()
        for origin_param_name in u_5_state_dict:
            weight_name, bias_name = get_init_layer_name(u_5_state_dict[origin_param_name].size())
            if weight_name:
                u_5_state_dict[origin_param_name] = 0.5*saved_dict[weight_name] + 0.5*u_5_state_dict[origin_param_name]
                u_5_state_dict[origin_param_name.replace('weight', 'bias')] = 0.5*saved_dict[bias_name] + 0.5*u_5_state_dict[origin_param_name.replace('weight', 'bias')]
        self.u_5.load_state_dict(u_5_state_dict)
        

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SpatialAttentionMapBlock_v1(nn.Module):
    '''
    @date: 2023.03.20
    @description: a block for getting Spatial Attention Map of SAOL (Spatially Attentive Output Layer)
    '''
    def __init__(self, h, w, in_c, mid_c):
        '''
        @params: h: height of feature map
                 w: width  of feature map
        '''
        super().__init__()

        self.sam_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=(3, 2), padding='same'),
            BatchNormReLU(mid_c),
            nn.Conv2d(in_channels=mid_c, out_channels=1, kernel_size=(3, 2), padding='same')
        )

    def forward(self, h):
        h = self.sam_block(h)
        # 扁平化输入张量, flatten the 2d feature map to 1d
        flat_input_tensor = h.reshape(h.shape[0], h.shape[1], -1)
        # 使用dim=2进行softmax
        softmax_output = nn.Softmax(dim=2)(flat_input_tensor)
        # 把形状变回 h 的样子
        sam = softmax_output.reshape(h.shape)
        return sam

class SpatialLogitsBlock_v1(nn.Module):
    '''
    @date: 2023.03.20
    @description: a block for getting Spatial Logits of SAOL (Spatially Attentive Output Layer)
    '''
    def __init__(self, h, w, in_c, mid_c, out_c):
        super().__init__()
        assert type(in_c) == tuple and len(in_c) == 3
        
        self.resize_1_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[0], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.resize_2_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[1], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.resize_3_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[2], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.sl_out_block = nn.Sequential(
            nn.BatchNorm2d(mid_c*3),
            nn.Conv2d(in_channels=mid_c*3, out_channels=out_c, kernel_size=(3, 2), padding='same'),
            nn.Softmax(dim=1)
        )

    def forward(self, h1, h2, h3):
        h1 = self.resize_1_block(h1)
        h2 = self.resize_2_block(h2)
        h3 = self.resize_3_block(h3) # h1 h2 h3 output shape: (batch_size, mid_c, h, w)
        cat_h = torch.cat((h1, h2, h3), dim=1) # cat_h shape: (batch_size, 3*mid_c, h, w)
        sl = self.sl_out_block(cat_h)
        return sl

def get_init_layer_name(weight_shape):
    weightshape2layername = {torch.Size([16, 16, 1, 3]): 'conv_3.1.weight',
                             torch.Size([16, 16, 3, 1]): 'conv_3.2.weight',
                             torch.Size([32, 32, 1, 3]): 'conv_3.7.weight',
                             torch.Size([32, 32, 3, 1]): 'conv_3.8.weight',
                             torch.Size([64, 64, 1, 3]): 'conv_3.13.weight',
                             torch.Size([64, 64, 3, 1]): 'conv_3.14.weight',
                             torch.Size([128, 128, 1, 3]): 'conv_3.19.weight',
                             torch.Size([128, 128, 3, 1]): 'conv_3.20.weight',
                             torch.Size([16, 16, 1, 5]): 'conv_5.1.weight',
                             torch.Size([16, 16, 5, 1]): 'conv_5.2.weight',
                             torch.Size([32, 32, 1, 5]): 'conv_5.7.weight',
                             torch.Size([32, 32, 5, 1]): 'conv_5.8.weight',
                             torch.Size([64, 64, 1, 5]): 'conv_5.13.weight',
                             torch.Size([64, 64, 5, 1]): 'conv_5.14.weight',
                             torch.Size([128, 128, 1, 5]): 'conv_5.19.weight',
                             torch.Size([128, 128, 5, 1]): 'conv_5.20.weight'}
    if weight_shape in weightshape2layername.keys():
        # print(f"weight_shape: {weight_shape} is in weightshape2layername.keys()")
        layerweightname = weightshape2layername[weight_shape]
        layerbiasname = layerweightname.replace('weight', 'bias')
        return layerweightname, layerbiasname
    else:
        # print(f"weight_shape: {weight_shape} is not in weightshape2layername.keys()")
        return False, False

class AttentionPool_v1(nn.Module):
    '''
    @description: attention & pool on width or height
                  similar to maxpool which select max in kernel, attention pool select max in global
    '''
    def __init__(self, attention_dim, attention_len, out_w, out_h, reduction_dim):
        '''
        @param attention_dim: 2 or 3, 2 means attention & pool on width, 3 means attention & pool on height
        '''
        super().__init__()
        assert attention_dim in [2,3]
        self.attention_dim = attention_dim
        self.out_w = out_w
        self.out_h = out_h
        self.attention_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(attention_len, reduction_dim),
            nn.BatchNorm1d(reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, attention_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        assert len(x.size()) == 4, "fit for 4-dim [N, C, W, H] tensor"
        if self.attention_dim == 2:
            # assert self.out_h == x.size()[3]
            attention_vector = torch.permute(x, (0, 2, 1, 3))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_w, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :, max_values_sorted[i], :] for i in range(x.size()[0])], dim=0)
        else:
            # assert self.out_w == x.size()[2]
            attention_vector = torch.permute(x, (0, 3, 1, 2))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_h, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :, :, max_values_sorted[i]] for i in range(x.size()[0])], dim=0)
        return pooled_x


class AttentionPool_v2(nn.Module):
    '''
    @description: attention & pool on width or height
                  similar to maxpool which select max in kernel, attention pool select max in global
    '''
    def __init__(self, attention_dim, attention_len, out_w, out_h, reduction_dim):
        '''
        @param attention_dim: 2 or 3, 2 means attention & pool on width, 3 means attention & pool on height
        '''
        super().__init__()
        assert attention_dim in [2,3]
        self.attention_dim = attention_dim
        self.out_w = out_w
        self.out_h = out_h
        self.attention_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(attention_len, reduction_dim),
            nn.BatchNorm1d(reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, attention_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert len(x.size()) == 4, "fit for 4-dim [N, C, W, H] tensor"
        if self.attention_dim == 2:
            # assert self.out_h == x.size()[3]
            attention_vector = torch.permute(x, (0, 2, 1, 3))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_w, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :, max_values_sorted[i], :] for i in range(x.size()[0])], dim=0)
        else:
            # assert self.out_w == x.size()[2]
            attention_vector = torch.permute(x, (0, 3, 1, 2))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_h, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :, :, max_values_sorted[i]] for i in range(x.size()[0])], dim=0)
        return pooled_x

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)) # type: ignore
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) # type: ignore
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FireSEBlock_v1(nn.Module):
    '''
    @date: 2023.04.12
    @description: FireBlock_v3 + SEBlock_v1
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        self.fire_1 = FireBlock_v3(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels,
                                   groups=groups)
        self.seblock_1 = SEBlock_v1(h_channels=input_channels, reduction=squeeze_channels)

    def forward(self, x):
        x = self.fire_1(x)
        x = self.seblock_1(x)
        return x

class LargeKernelFireSEBlock_v1(nn.Module):
    '''
    @date: 2023.04.12
    @description: LargeKernelFireBlock_v1 + SEBlock_v1
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v1(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels,
                                   groups=groups)
        self.seblock_1 = SEBlock_v1(h_channels=input_channels, reduction=squeeze_channels)

    def forward(self, x):
        x = self.fire_1(x)
        x = self.seblock_1(x)
        return x

class LargeKernelFireSEBlock_v2(nn.Module):
    '''
    @date: 2023.04.12
    @description: LargeKernelFireBlock_v2 + SEBlock_v3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v2(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels,
                                   groups=groups)
        self.seblock_1 = SEBlock_v3(h_channels=input_channels, reduction=squeeze_channels)

    def forward(self, x):
        x = self.fire_1(x)
        x = self.seblock_1(x)
        return x

class LargeKernelCABlock_v1(nn.Module):
    '''
    @date: 2023.04.13
    @description: LargeKernelFireBlock_v3 + SEBlock_v3 + residual connection
                  用LargeKernelFireBlock_v3的LN结果不太好，
                  similar to LargeKernelFireSEBlock, but for distinct to LargeKernelSABlock,
                  rename it to LargeKernelCABlock
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v3(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels)
        self.seblock_1 = SEBlock_v3(h_channels=input_channels, reduction=squeeze_channels)
        self.one = nn.Parameter(0.9999 * torch.ones(1)) # type: ignore
        self.zero = nn.Parameter(1E-4 * torch.ones(1)) # type: ignore

    def forward(self, x):
        h = self.fire_1(x)
        h = self.seblock_1(h)
        return (x.mul_(self.zero)).add_(h.mul_(self.one))


class LargeKernelCABlock_v2(nn.Module):
    '''
    @date: 2023.04.15
    @description: LargeKernelFireBlock_v4 + SEBlock_v3 + residual connection
                  similar to LargeKernelFireSEBlock, but for distinct to LargeKernelSABlock,
                  rename it to LargeKernelCABlock
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v4(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels)
        self.seblock_1 = SEBlock_v3(h_channels=input_channels, reduction=squeeze_channels)
        self.one = nn.Parameter(0.9999 * torch.ones(1)) # type: ignore
        self.zero = nn.Parameter(1E-4 * torch.ones(1)) # type: ignore

    def forward(self, x):
        h = self.fire_1(x)
        h = self.seblock_1(h)
        return (x.mul(self.zero)).add_(h.mul(self.one))

class LargeKernelSABlock_v1(nn.Module):
    '''
    @date: 2023.04.13
    @description: LargeKernelFireBlock_v3 + PRMLayer + residual connection
                  用LargeKernelFireBlock_v3的LN结果不太好，
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, head_num=1, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v3(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels)
        self.prm_1 = PRMLayer(groups=head_num)
        self.one = nn.Parameter(0.9999 * torch.ones(1)) # type: ignore
        self.zero = nn.Parameter(1E-4 * torch.ones(1)) # type: ignore

    def forward(self, x):
        h = self.fire_1(x)
        h = self.prm_1(h)
        return (x.mul_(self.zero)).add_(h.mul_(self.one))

class LargeKernelSABlock_v2(nn.Module):
    '''
    @date: 2023.04.15
    @description: LargeKernelFireBlock_v4 + PRMLayer + residual connection
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, head_num=1, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v4(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   e_5_channels=e_5_channels,
                                   groups=groups)
        self.prm_1 = PRMLayer(groups=head_num)
        self.one = nn.Parameter(0.9999 * torch.ones(1)) # type: ignore
        self.zero = nn.Parameter(1E-4 * torch.ones(1)) # type: ignore

    def forward(self, x):
        h = self.fire_1(x)
        h = self.prm_1(h)
        return (x.mul(self.zero)).add_(h.mul(self.one))

class LargeKernelSABlock_v3(nn.Module):
    '''
    @date: 2023.04.16
    @description: LargeKernelFireBlock_v5 + PRMLayer + residual connection
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, head_num=1, groups=1):
        super().__init__()
        self.fire_1 = LargeKernelFireBlock_v5(input_channels=input_channels,
                                   squeeze_channels=squeeze_channels,
                                   e_1_channels=e_1_channels,
                                   e_3_channels=e_3_channels,
                                   groups=groups)
        self.prm_1 = PRMLayer(groups=head_num)
        self.one = nn.Parameter(0.9999 * torch.ones(1)) # type: ignore
        self.zero = nn.Parameter(1E-4 * torch.ones(1)) # type: ignore

    def forward(self, x):
        h = self.fire_1(x)
        h = self.prm_1(h)
        return (x.mul(self.zero)).add_(h.mul(self.one))

class CASABlock_v1(nn.Module):
    '''
    @date: 2023.04.13
    @description: LargeKernelFireBlock_v3 + PRMLayer + residual connection
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, head_num=1, groups=1):
        super().__init__()
        self.ca_block_1 = LargeKernelCABlock_v1(input_channels=input_channels, squeeze_channels=squeeze_channels, e_1_channels=e_1_channels, e_3_channels=e_3_channels, e_5_channels=e_5_channels)
        self.sa_block_1 = LargeKernelSABlock_v1(input_channels=input_channels, squeeze_channels=squeeze_channels, e_1_channels=e_1_channels, e_3_channels=e_3_channels, e_5_channels=e_5_channels, head_num=head_num)

    def forward(self, x):
        x = self.ca_block_1(x)
        x = self.sa_block_1(x)
        return x

class Conv1dMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm1d(dim)
        self.a = nn.Sequential(
                nn.Conv1d(dim, dim, 1),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=5, dilation=3, padding="same", groups=dim)
        )

        self.v = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x

class LargeKernelFireBlock1D_v1(nn.Module):
    '''
    @date: 2023.05.13
    @description: squeeze and expand, designed for not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  use large kernel
                  replace LN in LargeKernelFireBlock_v2 with BN
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups=1):
        super().__init__()
        # squeeze
        self.squeeze_bn = nn.BatchNorm1d(input_channels)
        self.squeeze = nn.Conv1d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=1, padding='same', groups=groups)
        self.squeeze_act = nn.GELU()
        # expand
        self.expand_1 = nn.Conv1d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=1, padding='same', groups=groups)
        self.expand_3 = nn.Conv1d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=3, padding='same', groups=groups)
        self.expand_5 = nn.Conv1d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=5, padding='same', dilation=3, groups=groups)
        self.expand_act = nn.GELU()

    def forward(self, x):
        s = self.squeeze_bn(x)
        s = self.squeeze(s)
        s = self.squeeze_act(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.expand_act(x)
        return x

class SliceLayer1D(nn.Module):
    '''
    @date: 2023.05.15
    @description: 均匀的把输入序列切割成slice_num份，每份长度为slice_len
    '''
    def __init__(self, slice_len, slice_num):
        super().__init__()
        self.slice_len = slice_len
        self.slice_num = slice_num

    def forward(self, x):
        step = torch.div((x.shape[2] - self.slice_len), (self.slice_num - 1.), rounding_mode='floor').int()
        slice_list = list()
        for i in range(self.slice_num-1):
            # print(i*step, i*step+self.slice_len)
            slice_list.append(x[:, :, i*step:i*step+self.slice_len])
        slice_list.append(x[:, :, -self.slice_len:])
        x = torch.stack(slice_list, dim=2)
        x = x.reshape([x.shape[0], -1, self.slice_len])
        return x

if __name__ == "__main__":
    # model = LargeKernelFireBlock_v2(input_channels=64,
    #                                squeeze_channels=16,
    #                                e_1_channels=16,
    #                                e_3_channels=32,
    #                                e_5_channels=16,
    #                                groups=1)

    # print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = SliceLayer1D(512, 4)
    input = torch.randn(2, 1, 996)
    output = model(input)
    print(output.shape)
