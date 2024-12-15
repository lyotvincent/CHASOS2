

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from pretrained_model.custom_layer import *



class FineTunedAnchorModel_v20230330_vAddALayer(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
    @thop.profile: macs=3877713088.0, params=2009282.0
    @model parameter number: 2140354
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_4 = SKBlock_v2(h_channels=256, out_channels=256, reduction=128)
        self.squeeze_convolution_4 = nn.Sequential(
            FireBlock_v3(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.sk_block_4(h)              # output shape: (batch_size, 256, 16, 2)
        h = self.squeeze_convolution_4(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h


if __name__ == "__main__":
    model = FineTunedAnchorModel_v20230330_vAddALayer()

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
