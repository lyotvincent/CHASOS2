

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from parameters import GLOVE_PATH
from positional_encoding import PositionalEncoding
from pretrained_model.custom_layer import *
from ref_block.cc_attention import CrissCrossAttention, CrissCrossAttention_v2
from ref_block.flash_pytorch import FLASH, FLASHTransformer



class FineTunedOCRModel_v20230330_vAddALayer(nn.Module):
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
            nn.Linear(64, 1)
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

class FineTunedOCRModel_v20230507_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=114835520.0, params=298898.0
    @model parameter number: 315282
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 166, 16)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166, 16)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 128, 83, 8)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 128, 83, 8)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 256, 42, 4)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 256, 42, 4)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class OCRModel_v20230511_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 + 3*cca_v2加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=114835488.0, params=298865.0
    @model parameter number: 445850
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.cca_1 = CrissCrossAttention_v2(in_dim=64, sequeeze_scale=4)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.cca_2 = CrissCrossAttention_v2(in_dim=128, sequeeze_scale=4)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.cca_3 = CrissCrossAttention_v2(in_dim=256, sequeeze_scale=4)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 166, 16)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166, 16)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 128, 83, 8)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 128, 83, 8)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 256, 42, 4)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 256, 42, 4)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class OCRModel_v20230512_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 + 3*cca_v2加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=10740896.0, params=43313.0
    @model parameter number: 80309
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16

        self.flashtranformer = FLASHTransformer(
            dim = EMBEDDING_DIM, # embedding dimension of each token
            num_tokens = 4**5, # embedding dictionary size
            depth = 8,
            group_size = 256,
            query_key_dim = 128,
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(996, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, kmer_data):
        h = self.flashtranformer(kmer_data)
        h = self.out_layer(h)
        return h

class OCRModel_v20230512_v2(nn.Module):
    """
    @description: use no slice 3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=114835520.0, params=298898.0
    @model parameter number: 315282
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 16)
        h = self.first_conv(h)                      # output shape: (batch_size, 64, 996, 16)
        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 996, 16)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 498, 8)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 498, 8)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 249, 4)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 249, 4)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230513_v1(nn.Module):
    """
    @description: use conv1d, no slice 3* LargeKernelFireBlock1D_v1 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=42589216.0, params=138257.0
    @model parameter number: 154641
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)
        h = h.transpose(1, 2)                       # output shape: (batch_size, 16, 996)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 996
        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 996)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 498)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 498)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 249)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 249)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230513_v2(nn.Module):
    """
    @description: use conv1d, no slice 3* LargeKernelFireBlock1D_v1 + 3* conv1dmod 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=129982240.0, params=401233.0
    @model parameter number: 417617
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.conv1dmod_1 = Conv1dMod(64)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.conv1dmod_2 = Conv1dMod(128)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.conv1dmod_3 = Conv1dMod(256)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)
        h = h.transpose(1, 2)                       # output shape: (batch_size, 16, 996)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 996
        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 996)
        h = self.conv1dmod_1(h)                  # output shape: (batch_size, 64, 996)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 498)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 498)
        h = self.conv1dmod_2(h)                  # output shape: (batch_size, 128, 498)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 249)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 249)
        h = self.conv1dmod_3(h)                  # output shape: (batch_size, 256, 249)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class FineTunedOCRModel_v20230330_vAddALayer_ablation1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
    @thop.profile: macs=691585152.0, params=2009217.0
    @model parameter number: 2025601
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 2), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 2), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 2), padding=(1,0))
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
            nn.Linear(64, 1)
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

class FineTunedOCRModel_v20230330_vAddALayer_ablation2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
    @thop.profile: macs=981099520.0, params=1566689.0
    @model parameter number: 1583073
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v2(h_channels=1, out_channels=32, reduction=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.scale_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=(1,0)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.sk_block_4 = SKBlock_v2(h_channels=256, out_channels=256, reduction=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 16)
        h = self.scale_conv_1(h)   # output shape: (batch_size, 64, 498, 8)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 498, 8)
        h = self.scale_conv_2(h)   # output shape: (batch_size, 128, 249, 4)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 249, 4)
        h = self.scale_conv_3(h)   # output shape: (batch_size, 256, 125, 2)
        h = self.sk_block_4(h)              # output shape: (batch_size, 256, 125, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class FineTunedOCRModel_v20230330_vAddALayer_ablation3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
    @thop.profile: macs=189860672.0, params=290275.0
    @model parameter number: 306659
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock_v15(input_channels=1, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_32_channels=8)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v15(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_32_channels=16)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v15(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_32_channels=32)
        self.scale_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=(1,0)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_4 = LargeKernelFireBlock_v15(input_channels=256, squeeze_channels=64, e_1_channels=64, e_31_channels=128, e_32_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

        h = self.fireBlock_1(h)              # output shape: (batch_size, 32, 996, 16)
        h = self.scale_conv_1(h)   # output shape: (batch_size, 64, 498, 8)
        h = self.fireBlock_2(h)              # output shape: (batch_size, 64, 498, 8)
        h = self.scale_conv_2(h)   # output shape: (batch_size, 128, 249, 4)
        h = self.fireBlock_3(h)              # output shape: (batch_size, 128, 249, 4)
        h = self.scale_conv_3(h)   # output shape: (batch_size, 256, 125, 2)
        h = self.fireBlock_4(h)              # output shape: (batch_size, 256, 125, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class FineTunedOCRModel_v20230330_vAddALayer_ablation4(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
    @thop.profile: macs=550886720.0, params=283715.0
    @model parameter number: 300099
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock_v15(input_channels=1, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_32_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v15(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_32_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=(1,0)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v15(input_channels=256, squeeze_channels=64, e_1_channels=64, e_31_channels=128, e_32_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

        h = self.fireBlock_1(h)              # output shape: (batch_size, 32, 996, 16)
        h = self.scale_conv_1(h)   # output shape: (batch_size, 64, 498, 8)
        h = self.fireBlock_2(h)              # output shape: (batch_size, 64, 498, 8)
        h = self.scale_conv_2(h)   # output shape: (batch_size, 128, 249, 4)
        h = self.fireBlock_3(h)              # output shape: (batch_size, 128, 249, 4)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class FineTunedOCRModel_v20230330_vAddALayer_ablation5(nn.Module):
    """
    @description: 3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
    @thop.profile: macs=599907648.0, params=305219.0
    @model parameter number: 321603
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=1, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=(1,0)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

        h = self.fireBlock_1(h)              # output shape: (batch_size, 32, 996, 16)
        h = self.scale_conv_1(h)   # output shape: (batch_size, 64, 498, 8)
        h = self.fireBlock_2(h)              # output shape: (batch_size, 64, 498, 8)
        h = self.scale_conv_2(h)   # output shape: (batch_size, 128, 249, 4)
        h = self.fireBlock_3(h)              # output shape: (batch_size, 128, 249, 4)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class OCRModel_v20230514_v1(nn.Module):
    """
    @description: use conv1d, slice + 3* LargeKernelFireBlock1D_v1 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=8755232.0, params=148497.0
    @model parameter number: 164881
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 996, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230514_v2(nn.Module):
    """
    @description: use conv1d, slice + 3* LargeKernelFireBlock1D_v1 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=27558048.0, params=529857.0
    @model parameter number: 546241
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 996, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230514_v3(nn.Module):
    """
    @description: use conv1d, slice + 4* LargeKernelFireBlock1D_v1 加 3 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=15665312.0, params=542897.0
    @model parameter number: 559281
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.scale_conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
        )
        self.fireBlock_4 = LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 996, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.scale_conv_3(h)                  # output shape: (batch_size, 512, 21)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 512, 21)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v1(nn.Module):
    """
    @description: use conv1d, slice3*498 + 3* LargeKernelFireBlock1D_v1 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=20679072.0, params=136881.0
    @model parameter number: 153265
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=48, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1),
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(3):
            slice_list.append(h[:, i*249:i*249+498, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 996, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 498)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 249)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 249)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 125)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 125)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v2(nn.Module):
    """
    @description: use conv1d, slice3*498 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=17602976.0, params=105777.0
    @model parameter number: 122161
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=48, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=249, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=125, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(3):
            slice_list.append(h[:, i*249:i*249+498, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 3, 498, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 498)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 249)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 249)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 125)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 125)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v3(nn.Module):
    """
    @description: use conv1d, slice_11*166 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=6320288.0, params=108081.0
    @model parameter number: 124465
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=83, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=42, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 166, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v4(nn.Module):
    """
    @description: use conv1d, slice_11*166 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=23808160.0, params=406209.0
    @model parameter number: 422593
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=83, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=42, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 166, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v5(nn.Module):
    """
    @description: use conv1d, slice_11*166 + 4* LargeKernelFireBlock1D_v1 加 3 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=13252768.0, params=412977.0
    @model parameter number: 429361
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=176, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=83, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=42, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.scale_conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=21, slice_num=4)
        )
        self.fireBlock_4 = LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])
        h = torch.stack(slice_list, dim=1)       # output shape: (batch_size, 11, 996, 16)
        h = h.transpose(2, 3)
        h = h.reshape([h.shape[0], -1, h.shape[3]])

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42)
        h = self.scale_conv_3(h)                  # output shape: (batch_size, 512, 21)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 512, 21)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v6(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=18227232.0, params=106065.0
    @model parameter number: 122449
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v7(nn.Module):
    """
    @description: use conv1d, 5* LargeKernelFireBlock1D_v1 加 4 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=10415904.0, params=108165.0
    @model parameter number: 124549
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=16, squeeze_channels=4, e_1_channels=4, e_3_channels=8, e_5_channels=4)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1),
            SliceLayer1D(slice_len=512, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_4 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4)
        )
        self.fireBlock_5 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)                 # output shape: (batch_size, 16, 996)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 16, 996)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 32, 512)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 32, 512)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 64, 256)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 64, 256)
        h = self.slice_layer_3(h)                  # output shape: (batch_size, 128, 128)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 128, 128)
        h = self.slice_layer_4(h)                  # output shape: (batch_size, 256, 64)
        h = self.fireBlock_5(h)                 # output shape: (batch_size, 256, 64)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v8(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 6* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=33300512.0, params=193537.0
    @model parameter number: 209921
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
        )
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
        )
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v9(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D + biGRU试一下
                  use 16 embedding dim
    @thop.profile: macs=69410848.0, params=304209.0
    @model parameter number: 320593
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.gru = nn.GRU(128, 128, bidirectional=True)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.gru(h)[0]                         # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230515_v10(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D + biGRU试一下
                  use 16 embedding dim
    @thop.profile: macs=47718432.0, params=221777.0
    @model parameter number: 239217
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.transformer_1 = FLASH(dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.transformer_1(h)                        # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230520_v1(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 8* LargeKernelFireBlock1D_v1 加 3 * SliceLayer1D试一下
                  use 16 embedding dim
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        )
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        )
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        )
        self.slice_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4)
        )
        self.fireBlock_4 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.slice_layer_3(h)                  # output shape: (batch_size, 512, 64)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 512, 64)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230520_v2(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 4* LargeKernelFireBlock1D_v1 加 3 * SliceLayer1D试一下
                  use 16 embedding dim
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
        )
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
        )
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
        )
        self.slice_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4)
        )
        self.fireBlock_4 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.slice_layer_3(h)                  # output shape: (batch_size, 512, 64)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 512, 64)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230520_v3(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 4* LargeKernelFireBlock1D_v1 加 3 * SliceLayer1D试一下
                  use 16 embedding dim
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
        )
        self.convmod1d_1 = Conv1dMod(dim=64)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
        )
        self.convmod1d_2 = Conv1dMod(dim=128)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
        )
        self.convmod1d_3 = Conv1dMod(dim=256)
        self.slice_layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4)
        )
        self.fireBlock_4 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
        )
        self.convmod1d_4 = Conv1dMod(dim=512)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.convmod1d_1(h)                     # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.convmod1d_2(h)                     # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.convmod1d_3(h)                     # output shape: (batch_size, 256, 128)
        h = self.slice_layer_3(h)                  # output shape: (batch_size, 512, 64)
        h = self.fireBlock_4(h)                 # output shape: (batch_size, 512, 64)
        h = self.convmod1d_4(h)                     # output shape: (batch_size, 512, 64)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230524_v1(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
                  use 16 embedding dim
    @thop.profile: macs=18737152.0, params=617208.0
    @model parameter number: 633592
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1000)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

class OCRModel_v20230526_v1(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 6* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D + 2* GRU试一下
                  use 16 embedding dim
    @thop.profile: macs=161081344.0, params=936872.0
    @model parameter number: 953256
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.fireBlock_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        )
        self.slice_layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4)
        )
        self.fireBlock_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        )
        self.slice_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4)
        )
        self.fireBlock_3 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        )
        self.gru_0 = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1000)
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        h = h.transpose(1, 2)
        h = self.slice_layer_0(h)                  # output shape: (batch_size, 64, 512)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 512)
        h = self.slice_layer_1(h)                  # output shape: (batch_size, 128, 256)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 256)
        h = self.slice_layer_2(h)                  # output shape: (batch_size, 256, 128)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 128)
        h, _ = self.gru_0(h)                         # output shape: (batch_size, 256, 128)
        h = self.out_layer(h)                      # output shape: (batch_size, 1)
        return h

if __name__ == "__main__":
    model = OCRModel_v20230524_v1().cuda()

    from thop import profile
    input = torch.randint(4**5, (1, 996)).cuda()
    macs, params = profile(model, inputs=(input, )) # type: ignore
    print(f"thop.profile: macs={macs}, params={params}")

    print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


