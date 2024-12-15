'''
@author: 孙嘉良
@description: generate pretrained layer:
              in_c & out_c = 32,64,128 kernel_size=(1, 3) & (3, 1) & (1, 5) & (5, 1)
              in_c & out_c = 16,32,64,128 kernel_size=(1, 3) & (3, 1) & (1, 5) & (5, 1)
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from parameters import GLOVE_PATH
from positional_encoding import PositionalEncoding


class PretrainedLayers(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
    @thop.profile: macs=3840937152.0, params=667010.0
    @model parameter number: 798082
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), padding='same'), # (batch_size, 16, 996, 128)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), padding='same'), # (batch_size, 16, 996, 128)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 1), padding='same'), # (batch_size, 16, 996, 128)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,0)), # (batch_size, 16, 250, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'), # (batch_size, 32, 250, 32)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), padding='same'), # (batch_size, 32, 250, 32)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding='same'), # (batch_size, 32, 250, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(1,0)), # (batch_size, 32, 63, 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'), # (batch_size, 64, 63, 8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding='same'), # (batch_size, 64, 63, 8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'), # (batch_size, 64, 63, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(1,0)), # (batch_size, 64, 16, 2)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding='same'), # (batch_size, 128, 16, 2)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding='same'), # (batch_size, 128, 16, 2)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), padding='same') # (batch_size, 128, 16, 2)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1), padding='same'), # (batch_size, 16, 996, 128)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 5), padding='same'), # (batch_size, 16, 996, 128)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 1), padding='same'), # (batch_size, 16, 996, 128)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,0)), # (batch_size, 16, 250, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding='same'), # (batch_size, 32, 250, 32)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5), padding='same'), # (batch_size, 32, 250, 32)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), padding='same'), # (batch_size, 32, 250, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(1,0)), # (batch_size, 32, 63, 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'), # (batch_size, 64, 63, 8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), padding='same'), # (batch_size, 64, 63, 8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'), # (batch_size, 64, 63, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(1,0)), # (batch_size, 64, 16, 2)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding='same'), # (batch_size, 128, 16, 2)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), padding='same'), # (batch_size, 128, 16, 2)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 1), padding='same') # (batch_size, 128, 16, 2)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h3 = self.conv_3(h)              # output shape: (batch_size, 128, 16, 2)
        h5 = self.conv_5(h)              # output shape: (batch_size, 128, 16, 2)
        h = torch.cat([h3, h5], dim=1)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)            # output shape: (batch_size, 2)
        return h


if __name__=="__main__":
    model = PretrainedLayers()

    # from thop import profile
    # input = torch.randint(4**5, (1, 996))
    # macs, params = profile(model, inputs=(input, ))
    # print(f"thop.profile: macs={macs}, params={params}")

    # from torchstat import stat
    # stat(model, (1, 1000, 4))

    print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

