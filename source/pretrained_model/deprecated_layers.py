
import torch
import torch.nn as nn

def batchnorm_relu(h, channels):
    h = nn.BatchNorm2d(channels)(h)
    h = nn.ReLU()(h)
    return h


@DeprecationWarning
def fire_block(h, input_channels, squeeze_channels, expand_channels, groups):
    # SquueezeNet Fire Module
    # sequeeze
    squeeze_h = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)(h)
    squeeze_h = batchnorm_relu(squeeze_h, squeeze_channels)
    # expand
    # // expand_h_1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand_channels, kernel_size=(1, 1))(squeeze_h)
    # // expand_h_1 = nn.ReLU()(expand_h_1)
    # // expand_h_1 = nn.ReLU()(expand_h_1)
    expand_h_3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand_channels, kernel_size=(3, 3), padding='same', groups=groups)(squeeze_h)
    expand_h_3 = batchnorm_relu(expand_h_3, expand_channels)
    return expand_h_3

@DeprecationWarning
def se_block(h, h_channels, reduction):
    # squeeze
    z = nn.AdaptiveMaxPool2d((1, 1))(h)
    z = nn.Flatten()(z)
    # excitation
    s = nn.Linear(h_channels, reduction)(z)
    s = batchnorm_relu(s, reduction)
    s = nn.Linear(reduction, h_channels)(s)
    s = nn.Sigmoid()(nn.BatchNorm1d(reduction)(s))
    s = s.reshape(s.shape[0], s.shape[1], 1, 1)
    # scale
    x = torch.mul(h, s)
    return x


@DeprecationWarning
def sk_block(h, h_channels, out_channels, reduction):
    # 1 split
    u_1 = nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same')(h)
    u_1 = batchnorm_relu(u_1, out_channels)
    u_3 = nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')(h)
    u_3 = batchnorm_relu(u_3, out_channels)
    u_5 = nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same', dilation=2)(h)
    u_5 = batchnorm_relu(u_5, out_channels)
    # 2 fuse
    # 2.1 integrate information from all branches.
    u = u_1 + u_3 + u_5
    # 2.2 global average pooling.
    s = nn.AdaptiveMaxPool2d((1, 1))(u)
    s = nn.Flatten()(s)
    # 2.3 compact feature by simple fully connected (fc) layer.
    z = nn.Linear(u.shape[1], reduction)(s)
    z = batchnorm_relu(z, reduction)
    # 3 select
    # 3.1 Soft attention across channels
    u_a = nn.Linear(reduction, u.shape[1])(z)
    u_b = nn.Linear(reduction, u.shape[1])(z)
    u_c = nn.Linear(reduction, u.shape[1])(z)
    u_a, u_b, u_c = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
    # 3.2 The final feature map V is obtained through the attention weights on various kernels.
    v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
    return v
