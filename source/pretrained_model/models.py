'''
@author: 孙嘉良
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ref_block'))

import torch
import torch.nn as nn
from parameters import GLOVE_PATH
from positional_encoding import PositionalEncoding
from pretrained_model.custom_layer import *
from ref_block.PRMLayer import PRMLayer
from ref_block.cc_attention import CrissCrossAttention, CrissCrossAttention_v2



class PretrainedModel_v20230228(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with a simple MLP model
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.fc1 = nn.Linear(996*128, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        # self.fc1 = nn.Linear(332*128, 2048) # a 1000bp sequence has 332 stride=3 5-mers
        # self.fc2 = nn.Linear(2048, 4096)
        # self.fc3 = nn.Linear(4096, 1024)
        # self.fc4 = nn.Linear(1024, 128)
        # self.fc5 = nn.Linear(128, 2)

        # load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h)
        h = nn.Flatten()(h)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230302_v1(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with conv layers
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(4, 4), stride=4)
        self.fc1 = nn.Linear(31744, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        # load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]])
        h = self.conv1(h) # output shape: (batch_size, 16, 249, 32)
        h = nn.ReLU()(h)
        h = self.conv2(h) # output shape: (batch_size, 64, 62, 8)
        h = nn.ReLU()(h)
        h = nn.Flatten()(h) # output shape: (batch_size, 64*62*8)=(batch_size, 31744)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230302_v2(nn.Module):
    '''
    @description: this model is for testing the pretrained embedding with conv layers
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 4), stride=(3, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 4), stride=(3, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 4), stride=(3, 2))
        self.fc1 = nn.Linear(16896, 1024) # a 1000bp sequence has 996 stride=1 5-mers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding.state_dict()
        model_dict.update(embedding_dict)
        self.embedding.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding(h)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)
        h = self.conv1(h) # output shape: (batch_size, 16, 331, 63)
        h = nn.ReLU()(h)
        h = self.conv2(h) # output shape: (batch_size, 64, 109, 30)
        h = nn.ReLU()(h)
        h = self.conv3(h) # output shape: (batch_size, 128, 35, 14)
        h = nn.ReLU()(h)
        h = self.conv4(h) # output shape: (batch_size, 256, 11, 6)
        h = nn.ReLU()(h)
        h = nn.Flatten()(h) # output shape: (batch_size, 256*11*6)=(batch_size, 16896)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230306_v1(nn.Module):
    '''
    @description: first model combine 5-mers and 1-mers, but its preformance is not good.
                however, its anchor_acc achieve 1.00 in epoch 14, and ocr_acc achieve 0.96.
                to this extent, the model need to add pooling and resnet block.
    @Input: two seqs, one is 996 5-mers, the other is 1000 1-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=128)
        self.embedding_2 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_len=1000)
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 4), stride=(3, 2))
        self.conv_1_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 4), stride=(3, 2))
        self.conv_2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        self.conv_2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(3, 1))
        self.conv_2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(3, 1))
        self.fc1 = nn.Linear(19200, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data, monomer_data):
        # 5-mer
        h1 = self.embedding_1(kmer_data)
        h1 = self.pos_encoding(h1) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h1.reshape([-1, 1, h1.shape[1], h1.shape[2]]) # output shape: (batch_size, 1, 996, 128)
        h1 = self.conv_1_1(h1) # output shape: (batch_size, 16, 331, 63)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_2(h1) # output shape: (batch_size, 64, 109, 30)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_3(h1) # output shape: (batch_size, 128, 35, 14)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_4(h1) # output shape: (batch_size, 256, 11, 6)
        h1 = nn.ReLU()(h1)
        h1 = nn.Flatten()(h1) # output shape: (batch_size, 256*11*6)=(batch_size, 16896)
        # 1-mer
        h2 = self.embedding_2(monomer_data)
        h2 = self.pos_encoding(h2)
        h2 = h2.reshape([-1, 1, h2.shape[1], h2.shape[2]])
        h2 = self.conv_2_1(h2) # output shape: (batch_size, 16, 332, 3)
        h2 = nn.ReLU()(h2)
        h2 = self.conv_2_2(h2) # output shape: (batch_size, 32, 110, 1)
        h2 = nn.ReLU()(h2)
        h2 = self.conv_2_3(h2) # output shape: (batch_size, 64, 36, 1)
        h2 = nn.ReLU()(h2)
        h2 = nn.Flatten()(h2) # output shape: (batch_size, 64*36*1)=(batch_size, 2304)

        h = torch.cat((h1, h2), dim=1)
        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.ReLU()(h)
        h = self.fc4(h)
        h = nn.ReLU()(h)
        h = self.fc5(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230306_v2(nn.Module):
    '''
    @Input: two seqs, one is 996 5-mers, the other is 1000 1-mers
    此模型未成功运行
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=996)
        # self.embedding_2 = nn.Embedding(num_embeddings=4, embedding_dim=8)
        # self.embedding.requires_grad_(False)
        self.pos_encoding = PositionalEncoding(embedding_dim=996, max_seq_len=1000)
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_1 = nn.MaxPool2d((4, 4))
        self.conv_1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_2 = nn.MaxPool2d((4, 4))
        self.conv_1_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_3 = nn.MaxPool2d((4, 4))
        self.conv_1_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv_1_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.pool_1_4 = nn.MaxPool2d((4, 4))
        # self.conv_2_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 4), stride=(3, 2))
        # self.conv_2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(3, 1))
        # self.conv_2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(3, 1))
        self.fc1 = nn.Linear(256*3*3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        # 5-mer
        h1 = self.embedding_1(kmer_data)
        h1 = self.pos_encoding(h1) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h1.reshape([-1, 1, h1.shape[1], h1.shape[2]]) # output shape: (batch_size, 1, 996, 996)
        h1 = self.conv_1_1(h1) # output shape: (batch_size, 64, 996, 996)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_2(h1) # output shape: (batch_size, 64, 996, 996)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_1(h1) # output shape: (batch_size, 64, 249, 249)
        h1 = self.conv_1_3(h1) # output shape: (batch_size, 128, 249, 249)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_4(h1) # output shape: (batch_size, 128, 249, 249)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_2(h1) # output shape: (batch_size, 128, 62, 62)
        h1 = self.conv_1_5(h1) # output shape: (batch_size, 256, 62, 62)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_6(h1) # output shape: (batch_size, 256, 62, 62)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_3(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = self.conv_1_7(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = nn.ReLU()(h1)
        h1 = self.conv_1_8(h1) # output shape: (batch_size, 256, 15, 15)
        h1 = nn.ReLU()(h1)
        h1 = self.pool_1_4(h1) # output shape: (batch_size, 256, 3, 3)
        h1 = nn.Flatten()(h1) # output shape: (batch_size, 256*3*3)
        # 1-mer
        # h2 = self.embedding_2(monomer_data)
        # h2 = self.pos_encoding(h2)
        # h2 = h2.reshape([-1, 1, h2.shape[1], h2.shape[2]]) # output shape: (batch_size, 1, 1000, 8)
        # h2 = self.conv_2_1(h2) # output shape: (batch_size, 16, 332, 3)
        # h2 = nn.ReLU()(h2)
        # h2 = self.conv_2_2(h2) # output shape: (batch_size, 32, 110, 1)
        # h2 = nn.ReLU()(h2)
        # h2 = self.conv_2_3(h2) # output shape: (batch_size, 64, 36, 1)
        # h2 = nn.ReLU()(h2)
        # h2 = nn.Flatten()(h2) # output shape: (batch_size, 64*36*1)=(batch_size, 2304)

        # h = torch.cat((h1, h2), dim=1)
        h = self.fc1(h1)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230307(nn.Module):
    '''
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()
        self.sequential_2 = self.__get_sequential()
        self.sequential_3 = self.__get_sequential()
        self.sequential_4 = self.__get_sequential()
        self.sequential_5 = self.__get_sequential()
        self.sequential_6 = self.__get_sequential()
        self.sequential_7 = self.__get_sequential()
        self.sequential_8 = self.__get_sequential()
        self.sequential_9 = self.__get_sequential()
        self.sequential_10 = self.__get_sequential()
        self.sequential_11 = self.__get_sequential()

        self.fc1 = nn.Linear(2816, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        # 5-mer
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]])
        h1 = h[:, :, 0:166, :]
        h2 = h[:, :, 83:249, :]
        h3 = h[:, :, 166:332, :]
        h4 = h[:, :, 249:415, :]
        h5 = h[:, :, 332:498, :]
        h6 = h[:, :, 415:581, :]
        h7 = h[:, :, 498:664, :]
        h8 = h[:, :, 581:747, :]
        h9 = h[:, :, 664:830, :]
        h10 = h[:, :, 747:913, :]
        h11 = h[:, :, 830:996, :]

        h = torch.cat([self.sequential_1(h1), self.sequential_2(h2), self.sequential_3(h3), self.sequential_4(h4), self.sequential_5(h5), self.sequential_6(h6), self.sequential_7(h7), self.sequential_8(h8), self.sequential_9(h9), self.sequential_10(h10), self.sequential_11(h11)], dim=1)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230308(nn.Module):
    '''
    @description: stack 11 subgraphs model
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230309(nn.Module):
    '''
    @description: stack 11 subgraphs, with group convolution
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential()

        self.fc1 = nn.Linear(352, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=88, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=88, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=176, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=176, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230313(nn.Module):
    '''
    @description: stack 11 subgraphs, with group convolution * 2
                                         + random group convolution * 2
                                         + conv * 2
    @Input: a seq with 996 5-mers
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sequential_1 = self.__get_sequential_1()
        self.sequential_2 = self.__get_sequential_2()

        self.fc1 = nn.Linear(352, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def __get_sequential_1(self):
        # nn.GroupNorm(11, 88),
        # import torchvision.models.shufflenetv2
        return nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=88, out_channels=88, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )

    def __get_sequential_2(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.Conv2d(in_channels=176, out_channels=176, kernel_size=(5, 5), stride=(1, 1), padding='same', groups=11),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), padding=(2,2)),
            nn.Conv2d(in_channels=176, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=352, out_channels=352, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h) # output shape: (batch_size, seq_len(kmer number, 996), embedding_dim(128))
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]

        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sequential_1(h)
        h = channel_shuffle(h, 11)
        h = self.sequential_2(h)

        h = self.fc1(h)
        h = nn.ReLU()(h)
        h = self.fc2(h)
        h = nn.ReLU()(h)
        h = self.fc3(h)
        h = nn.Sigmoid()(h)
        return h

class PretrainedModel_v20230315(nn.Module):
    '''
    @description: Embedding + PositionalEncoding + (group FireBlock_v1+SEBlock_v1)*2 + SKBlock_v1 + DenseOutLayer
    @model parameter number: 502730
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.group_convolution = nn.Sequential(
            FireBlock_v1(input_channels=11, squeeze_channels=22, expand_channels=44, groups=11),
            SEBlock_v1(h_channels=44, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.shuffle_group_convolution = nn.Sequential(
            FireBlock_v1(input_channels=44, squeeze_channels=44, expand_channels=88, groups=11),
            SEBlock_v1(h_channels=88, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block = SKBlock_v1(h_channels=88, out_channels=176, reduction=44)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(176, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h)
        # stack
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]
        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        # group convolution
        h = self.group_convolution(h)
        # shuffle group convolution
        # h = channel_shuffle(h, 11)
        h = self.shuffle_group_convolution(h)
        # SKBlock_v1
        h = self.sk_block(h)
        h = self.out_layer(h)
        return h

class PretrainedModel_v20230318(nn.Module):
    '''
    @description: Embedding + PositionalEncoding + (FireBlock_v2+SEBlock_v1)*2 + SKBlock_v1 + DenseOutLayer
    @model parameter number: 502730
    '''
    def __init__(self):
        super().__init__()

        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=166)
        self.pos_encoding = PositionalEncoding(embedding_dim=166, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v1(h_channels=11, out_channels=88, reduction=44)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v2(input_channels=88, squeeze_channels=22, e_1_channels=44, e_3_channels=44, groups=1),
            SEBlock_v1(h_channels=88, reduction=22),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block_2 = SKBlock_v1(h_channels=88, out_channels=176, reduction=44)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v2(input_channels=176, squeeze_channels=44, e_1_channels=88, e_3_channels=88, groups=1),
            SEBlock_v1(h_channels=176, reduction=44),
            nn.MaxPool2d((4, 4), padding=(2,2))
        )
        self.sk_block_3 = SKBlock_v1(h_channels=176, out_channels=352, reduction=44)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(352, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding_dim166.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, kmer_data):
        h = self.embedding_1(kmer_data)
        h = self.pos_encoding(h)
        # stack
        h1 = h[:, 0:166, :]
        h2 = h[:, 83:249, :]
        h3 = h[:, 166:332, :]
        h4 = h[:, 249:415, :]
        h5 = h[:, 332:498, :]
        h6 = h[:, 415:581, :]
        h7 = h[:, 498:664, :]
        h8 = h[:, 581:747, :]
        h9 = h[:, 664:830, :]
        h10 = h[:, 747:913, :]
        h11 = h[:, 830:996, :]
        h = torch.stack((h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11),dim=1)
        h = self.sk_block_1(h)
        h = self.squeeze_convolution_1(h)
        h = self.sk_block_2(h)
        h = self.squeeze_convolution_2(h)
        h = self.sk_block_3(h)
        h = self.out_layer(h)
        return h

class PretrainedModel_v20230319(nn.Module):
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
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230320(nn.Module):
    """
    @description: model for 996bp, no slice, and use n*1 & 1*n conv (Asymmetric Convolution)
                  add SAOL
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
        # Spatial Attention Map
        self.sam_layer = SpatialAttentionMapBlock_v1(h=16, w=2, in_c=256, mid_c=32)
        # Spatial Logits
        self.sl_layer = SpatialLogitsBlock_v1(h=16, w=2, in_c=(64, 128, 256), mid_c=16, out_c=2)

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

        h1 = self.sk_block_1(h)                 # output shape: (batch_size, 32, 996, 128)
        h1 = self.squeeze_convolution_1(h1)     # output shape: (batch_size, 64, 250, 32)
        h1 = self.sk_block_2(h1)                # output shape: (batch_size, 64, 250, 32)
        h2 = self.squeeze_convolution_2(h1)     # output shape: (batch_size, 128, 63, 8)
        h2 = self.sk_block_3(h2)                # output shape: (batch_size, 128, 63, 8)
        h3 = self.squeeze_convolution_3(h2)     # output shape: (batch_size, 256, 16, 2)

        sam = self.sam_layer(h3)                # output shape: (batch_size, 1, 16, 2)
        sl = self.sl_layer(h1, h2, h3)          # output shape: (batch_size, 2, 16, 2)
        # Spatial Weighted Sum
        sws = torch.sum(torch.mul(sam, sl), dim=(2,3))
        return sws

class PretrainedModel_v20230324(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  use new layer FireBlock_v4，SEBlock_v2，SKBlock_v3 which reduce some BN and ReLU
    @thop.profile: macs=3794374976.0, params=665442.0
    @model parameter number: 796514
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v3(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v4(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v2(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v3(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v4(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v2(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v3(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v4(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v2(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230325(nn.Module):
    """
    @description: model for 996bp, no slice, and use n*1 & 1*n conv (Asymmetric Convolution)
                  add more channels SAOL
    @thop.profile: macs=3842888224.0, params=711809.0
    @model parameter number: 842881
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
        # Spatial Attention Map
        self.sam_layer = SpatialAttentionMapBlock_v1(h=16, w=2, in_c=256, mid_c=32)
        # Spatial Logits
        self.sl_layer = SpatialLogitsBlock_v1(h=16, w=2, in_c=(64, 128, 256), mid_c=16, out_c=16)

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

        h1 = self.sk_block_1(h)                     # output shape: (batch_size, 32, 996, 128)
        h1 = self.squeeze_convolution_1(h1)         # output shape: (batch_size, 64, 250, 32)
        h1 = self.sk_block_2(h1)                    # output shape: (batch_size, 64, 250, 32)
        h2 = self.squeeze_convolution_2(h1)         # output shape: (batch_size, 128, 63, 8)
        h2 = self.sk_block_3(h2)                    # output shape: (batch_size, 128, 63, 8)
        h3 = self.squeeze_convolution_3(h2)         # output shape: (batch_size, 256, 16, 2)

        sam = self.sam_layer(h3)                    # output shape: (batch_size, 1, 16, 2)
        sl = self.sl_layer(h1, h2, h3)              # output shape: (batch_size, 16, 16, 2)
        # Spatial Weighted Sum
        weighted_spatial_sum = torch.mul(sam, sl)   # output shape: (batch_size, 16, 16, 2)
        weighted_spatial_sum = weighted_spatial_sum.reshape((weighted_spatial_sum.shape[0], 2, -1, weighted_spatial_sum.shape[2], weighted_spatial_sum.shape[3]))
        sws = torch.sum(weighted_spatial_sum, dim=(2,3,4))
        return sws

class PretrainedModel_v20230326(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                    ,replace BN with LN
    @thop.profile: macs=3840937152.0, params=51881954.0
    @model parameter number: 52013026
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v4(h_channels=1, out_channels=32, reduction=16, ln_size=(996, 128))
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v5(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, ln_size=(996, 128), groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v4(h_channels=64, out_channels=64, reduction=32, ln_size=(250, 32))
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v5(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, ln_size=(250, 32), groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v4(h_channels=128, out_channels=128, reduction=64, ln_size=(63, 8))
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v5(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, ln_size=(63, 8), groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230326_vSAOL(nn.Module):
    """
    @description: model for 996bp, no slice, and use n*1 & 1*n conv (Asymmetric Convolution)
                  add SAOL, and self-Distillation GAP-FC
                  based on 0320
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
        # Spatial Attention Map
        self.sam_layer = SpatialAttentionMapBlock_v1(h=16, w=2, in_c=256, mid_c=32)
        # Spatial Logits
        self.sl_layer = SpatialLogitsBlock_v1(h=16, w=2, in_c=(64, 128, 256), mid_c=16, out_c=2)
        # global average pooling (GAP) layer and fully-connected (FC) layers
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h1 = self.sk_block_1(h)                 # output shape: (batch_size, 32, 996, 128)
        h1 = self.squeeze_convolution_1(h1)     # output shape: (batch_size, 64, 250, 32)
        h1 = self.sk_block_2(h1)                # output shape: (batch_size, 64, 250, 32)
        h2 = self.squeeze_convolution_2(h1)     # output shape: (batch_size, 128, 63, 8)
        h2 = self.sk_block_3(h2)                # output shape: (batch_size, 128, 63, 8)
        h3 = self.squeeze_convolution_3(h2)     # output shape: (batch_size, 256, 16, 2)

        # Spatially Attentive Output Layer
        sam = self.sam_layer(h3)                # output shape: (batch_size, 1, 16, 2)
        sl = self.sl_layer(h1, h2, h3)          # output shape: (batch_size, 2, 16, 2)
        # Spatial Weighted Sum
        sws = torch.sum(torch.mul(sam, sl), dim=(2,3)) # output shape: (batch_size, 2)

        # GAP-FC
        gap_fc_out = self.out_layer(h3)                  # output shape: (batch_size, 2)
        return sws, gap_fc_out

class PretrainedModel_v20230327(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
                  use new SKBlock_v5, FireBlock_v6, which replace kernel 1*3 & dilation 2 with kernel 1*5
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v5(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v6(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v5(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v6(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v5(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v6(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        # // model_dict = self.embedding_1.state_dict()
        # // model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(embedding_dict)

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
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230328(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
                  init 1*3 3*1 weight bias with pretraind layer
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.sk_block_1 = SKBlock_v6(h_channels=1, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v7(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=32),
            nn.MaxPool2d((4, 4), padding=(2,0))
        )
        self.sk_block_2 = SKBlock_v6(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v7(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.sk_block_3 = SKBlock_v6(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v7(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            nn.MaxPool2d((4, 4), padding=(1,0))
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        # // model_dict = self.embedding_1.state_dict()
        # // model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(embedding_dict)

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
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230329_vAttentionPool(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
                  replace MaxPool2d with AttentionPool_v1
    @thop.profile: macs=3841239801.0, params=966462.0
    @model parameter number: 1097534
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
            AttentionPool_v1(attention_dim=3, attention_len=128, out_w=996, out_h=32, reduction_dim=32),
            AttentionPool_v1(attention_dim=2, attention_len=996, out_w=250, out_h=32, reduction_dim=128)
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            AttentionPool_v1(attention_dim=3, attention_len=32, out_w=250, out_h=8, reduction_dim=8),
            AttentionPool_v1(attention_dim=2, attention_len=250, out_w=63, out_h=8, reduction_dim=63)
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            AttentionPool_v1(attention_dim=3, attention_len=8, out_w=63, out_h=2, reduction_dim=2),
            AttentionPool_v1(attention_dim=2, attention_len=63, out_w=16, out_h=2, reduction_dim=16)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230330_vRes(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319
                  add residual connection
    @thop.profile: macs=3064390816.0, params=449970.0
    @model parameter number: 1890354
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.sk_block_1 = SKBlock_v2(h_channels=32, out_channels=32, reduction=16)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1),
            SEBlock_v1(h_channels=32, reduction=8)
        )
        self.pool_1 = nn.MaxPool2d((4, 4), padding=(2,0))
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=16)
        )
        self.pool_2 = nn.MaxPool2d((4, 4), padding=(1,0))
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=32)
        )
        # self.pool_3 = nn.MaxPool2d((4, 4), padding=(1,0))
        # self.sk_block_4 = SKBlock_v2(h_channels=256, out_channels=256, reduction=128)
        # self.squeeze_convolution_4 = nn.Sequential(
        #     FireBlock_v3(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
        #     SEBlock_v1(h_channels=256, reduction=64)
        # )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1_sk = self.sk_block_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1_sq = self.squeeze_convolution_1(h1_sk)   # output shape: (batch_size, 32, 996, 128)
        h1 = torch.cat([h1_sk, h1_sq], dim=1)       # output shape: (batch_size, 64, 996, 128)
        h1 = self.pool_1(h1)                        # output shape: (batch_size, 64, 250, 32)

        h2_sk = self.sk_block_2(h1)                 # output shape: (batch_size, 64, 250, 32)
        h2_sq = self.squeeze_convolution_2(h2_sk)   # output shape: (batch_size, 64, 250, 32)
        h2 = torch.cat([h2_sk, h2_sq], dim=1)       # output shape: (batch_size, 128, 250, 32)
        h2 = self.pool_2(h2)                        # output shape: (batch_size, 128, 63, 8)

        h3_sk = self.sk_block_3(h2)                 # output shape: (batch_size, 128, 63, 8)
        h3_sq = self.squeeze_convolution_3(h3_sk)   # output shape: (batch_size, 128, 63, 8)
        h3 = torch.cat([h3_sk, h3_sq], dim=1)       # output shape: (batch_size, 256, 63, 8)
        # // h3 = self.pool_3(h3)                        # output shape: (batch_size, 256, 16, 2)

        # // h4_sk = self.sk_block_4(h3)                 # output shape: (batch_size, 256, 16, 2)
        # // h4_sq = self.squeeze_convolution_4(h4_sk)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h3)                   # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230330_vAddALayer(nn.Module):
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

class PretrainedModel_v20230403_vAttentionPool(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319,
                  replace MaxPool2d with AttentionPool_v1
    @thop.profile: macs=3841239801.0, params=966462.0
    @model parameter number: 1097534
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
            AttentionPool_v1(attention_dim=3, attention_len=128, out_w=996, out_h=32, reduction_dim=32),
            AttentionPool_v1(attention_dim=2, attention_len=996, out_w=250, out_h=32, reduction_dim=128)
        )
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=32)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=64),
            AttentionPool_v1(attention_dim=3, attention_len=32, out_w=250, out_h=8, reduction_dim=8),
            AttentionPool_v1(attention_dim=2, attention_len=250, out_w=63, out_h=8, reduction_dim=63)
        )
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=64)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=128),
            AttentionPool_v1(attention_dim=3, attention_len=8, out_w=63, out_h=2, reduction_dim=2),
            AttentionPool_v1(attention_dim=2, attention_len=63, out_w=16, out_h=2, reduction_dim=16)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.sk_block_1(h)              # output shape: (batch_size, 32, 996, 128)
        h = self.squeeze_convolution_1(h)   # output shape: (batch_size, 64, 250, 32)
        h = self.sk_block_2(h)              # output shape: (batch_size, 64, 250, 32)
        h = self.squeeze_convolution_2(h)   # output shape: (batch_size, 128, 63, 8)
        h = self.sk_block_3(h)              # output shape: (batch_size, 128, 63, 8)
        h = self.squeeze_convolution_3(h)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)               # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230405_vRes(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230319
                  add residual connection, change pooling 4 to 2
    @thop.profile: macs=2337670560.0, params=433510.0
    @model parameter number: 564582
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1))
        self.sk_block_1 = SKBlock_v2(h_channels=16, out_channels=16, reduction=4)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=16, squeeze_channels=4, e_1_channels=4, e_3_channels=8, e_5_channels=4, groups=1),
            SEBlock_v1(h_channels=16, reduction=4)
        )
        self.pool_1 = nn.MaxPool2d((2, 2), padding=(0,0))
        self.sk_block_2 = SKBlock_v2(h_channels=32, out_channels=32, reduction=8)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1),
            SEBlock_v1(h_channels=32, reduction=8)
        )
        self.pool_2 = nn.MaxPool2d((2, 2), padding=(0,0))
        self.sk_block_3 = SKBlock_v2(h_channels=64, out_channels=64, reduction=16)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=16)
        )
        self.pool_3 = nn.MaxPool2d((2, 2), padding=(1,0))
        self.sk_block_4 = SKBlock_v2(h_channels=128, out_channels=128, reduction=32)
        self.squeeze_convolution_4 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=32)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.first_conv(h)                      # output shape: (batch_size, 16, 996, 128)

        h1_sk = self.sk_block_1(h)                  # output shape: (batch_size, 16, 996, 128)
        h1_sq = self.squeeze_convolution_1(h1_sk)   # output shape: (batch_size, 16, 996, 128)
        h1 = torch.cat([h1_sk, h1_sq], dim=1)       # output shape: (batch_size, 32, 996, 128)
        h1 = self.pool_1(h1)                        # output shape: (batch_size, 32, 498, 64)

        h2_sk = self.sk_block_2(h1)                 # output shape: (batch_size, 32, 498, 64)
        h2_sq = self.squeeze_convolution_2(h2_sk)   # output shape: (batch_size, 32, 498, 64)
        h2 = torch.cat([h2_sk, h2_sq], dim=1)       # output shape: (batch_size, 64, 498, 64)
        h2 = self.pool_2(h2)                        # output shape: (batch_size, 64, 249, 32)

        h3_sk = self.sk_block_3(h2)                 # output shape: (batch_size, 64, 249, 32)
        h3_sq = self.squeeze_convolution_3(h3_sk)   # output shape: (batch_size, 64, 249, 32)
        h3 = torch.cat([h3_sk, h3_sq], dim=1)       # output shape: (batch_size, 128, 249, 32)
        h3 = self.pool_3(h3)                        # output shape: (batch_size, 128, 125, 16)

        h4_sk = self.sk_block_4(h3)                 # output shape: (batch_size, 128, 125, 16)
        h4_sq = self.squeeze_convolution_4(h4_sk)   # output shape: (batch_size, 128, 125, 16)
        h4 = torch.cat([h4_sk, h4_sq], dim=1)       # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                   # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230406_vResAttentionPool(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  based on PretrainedModel_v20230405_vRes
                  add residual connection, pooling 2
    @thop.profile: macs=2384740224.0, params=1121554.0
    @model parameter number: 1252626
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 1))
        self.sk_block_1 = SKBlock_v2(h_channels=16, out_channels=16, reduction=4)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=16, squeeze_channels=4, e_1_channels=4, e_3_channels=8, e_5_channels=4, groups=1),
            SEBlock_v1(h_channels=16, reduction=4)
        )
        self.pool_11 = AttentionPool_v2(attention_dim=3, attention_len=128, out_w=996, out_h=64, reduction_dim=32)
        self.pool_12 = AttentionPool_v2(attention_dim=2, attention_len=996, out_w=512, out_h=64, reduction_dim=256)
        self.sk_block_2 = SKBlock_v2(h_channels=32, out_channels=32, reduction=8)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1),
            SEBlock_v1(h_channels=32, reduction=8)
        )
        self.pool_21 = AttentionPool_v2(attention_dim=3, attention_len=64, out_w=512, out_h=32, reduction_dim=16)
        self.pool_22 = AttentionPool_v2(attention_dim=2, attention_len=512, out_w=256, out_h=32, reduction_dim=128)
        self.sk_block_3 = SKBlock_v2(h_channels=64, out_channels=64, reduction=16)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=16)
        )
        self.pool_31 = AttentionPool_v2(attention_dim=3, attention_len=32, out_w=256, out_h=16, reduction_dim=8)
        self.pool_32 = AttentionPool_v2(attention_dim=2, attention_len=256, out_w=128, out_h=16, reduction_dim=64)
        self.sk_block_4 = SKBlock_v2(h_channels=128, out_channels=128, reduction=32)
        self.squeeze_convolution_4 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=32)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
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

        h = self.first_conv(h)                      # output shape: (batch_size, 16, 996, 128)

        h1_sk = self.sk_block_1(h)                  # output shape: (batch_size, 16, 996, 128)
        h1_sq = self.squeeze_convolution_1(h1_sk)   # output shape: (batch_size, 16, 996, 128)
        h1 = torch.cat([h1_sk, h1_sq], dim=1)       # output shape: (batch_size, 32, 996, 128)
        h1 = self.pool_11(self.pool_12(h1))         # output shape: (batch_size, 32, 512, 64)

        h2_sk = self.sk_block_2(h1)                 # output shape: (batch_size, 32, 512, 64)
        h2_sq = self.squeeze_convolution_2(h2_sk)   # output shape: (batch_size, 32, 512, 64)
        h2 = torch.cat([h2_sk, h2_sq], dim=1)       # output shape: (batch_size, 64, 512, 64)
        h2 = self.pool_21(self.pool_22(h2))         # output shape: (batch_size, 64, 256, 32)

        h3_sk = self.sk_block_3(h2)                 # output shape: (batch_size, 64, 256, 32)
        h3_sq = self.squeeze_convolution_3(h3_sk)   # output shape: (batch_size, 64, 256, 32)
        h3 = torch.cat([h3_sk, h3_sq], dim=1)       # output shape: (batch_size, 128, 256, 32)
        h3 = self.pool_31(self.pool_32(h3))         # output shape: (batch_size, 128, 128, 16)

        h4_sk = self.sk_block_4(h3)                 # output shape: (batch_size, 128, 128, 16)
        h4_sq = self.squeeze_convolution_4(h4_sk)   # output shape: (batch_size, 128, 128, 16)
        h4 = torch.cat([h4_sk, h4_sq], dim=1)       # output shape: (batch_size, 256, 128, 16)
        h = self.out_layer(h4)                   # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230407_vconvpool(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
    @thop.profile: macs=9686914176.0, params=1645354.0
    @model parameter number: 1967146
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.sk_block_1 = SKBlock_v2(h_channels=32, out_channels=32, reduction=8)
        self.squeeze_convolution_1 = nn.Sequential(
            FireBlock_v3(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1),
            SEBlock_v1(h_channels=32, reduction=8)
        )
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.sk_block_2 = SKBlock_v2(h_channels=64, out_channels=64, reduction=16)
        self.squeeze_convolution_2 = nn.Sequential(
            FireBlock_v3(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
            SEBlock_v1(h_channels=64, reduction=16)
        )
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.sk_block_3 = SKBlock_v2(h_channels=128, out_channels=128, reduction=32)
        self.squeeze_convolution_3 = nn.Sequential(
            FireBlock_v3(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
            SEBlock_v1(h_channels=128, reduction=32)
        )
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.sk_block_4 = SKBlock_v2(h_channels=256, out_channels=256, reduction=64)
        self.squeeze_convolution_4 = nn.Sequential(
            FireBlock_v3(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
            SEBlock_v1(h_channels=256, reduction=64)
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1_sk = self.sk_block_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1_sq = self.squeeze_convolution_1(h1_sk)   # output shape: (batch_size, 32, 996, 128)
        h1 = h1_sk+h1_sq                            # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2_sk = self.sk_block_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2_sq = self.squeeze_convolution_2(h2_sk)   # output shape: (batch_size, 64, 498, 64)
        h2 = h2_sk+h2_sq                            # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3_sk = self.sk_block_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3_sq = self.squeeze_convolution_3(h3_sk)   # output shape: (batch_size, 128, 249, 32)
        h3 = h3_sk+h3_sq                            # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4_sk = self.sk_block_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        # h4_sq = self.squeeze_convolution_4(h4_sk)   # output shape: (batch_size, 256, 125, 16)
        # h4 = h4_sk+h4_sq                            # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4_sk)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230412(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  dont use SKBlock
                  效果不行，在GM12878上大概88%左右
    @thop.profile: macs=2068173728.0, params=435394.0
    @model parameter number: 566466
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = FireSEBlock_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.fireSEBlock_2 = FireSEBlock_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.fireSEBlock_3 = FireSEBlock_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.fireSEBlock_4 = FireSEBlock_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.fireSEBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireSEBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.fireSEBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230412_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
    @thop.profile: macs=2198852512.0, params=457154.0
    @model parameter number: 588226
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = LargeKernelFireSEBlock_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.fireSEBlock_2 = LargeKernelFireSEBlock_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.fireSEBlock_3 = LargeKernelFireSEBlock_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.fireSEBlock_4 = LargeKernelFireSEBlock_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.fireSEBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireSEBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.fireSEBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

@DeprecationWarning
class PretrainedModel_v20230412_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
    @thop.profile: macs=2202424672.0, params=457154.0
    @model parameter number: 588247
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = LargeKernelFireSEBlock_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.prm_1 = PRMLayer(groups=1)
        self.fireSEBlock_2 = LargeKernelFireSEBlock_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.prm_2 = PRMLayer(groups=1)
        self.fireSEBlock_3 = LargeKernelFireSEBlock_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.prm_3 = PRMLayer(groups=1)
        self.fireSEBlock_4 = LargeKernelFireSEBlock_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireSEBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireSEBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.fireSEBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230413_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
                  replace all ReLU in PretrainedModel_v20230412_v3 with GELU
    @thop.profile: macs=2164168032.0, params=455954.0
    @model parameter number: 588007
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = LargeKernelFireSEBlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.prm_1 = PRMLayer(groups=1)
        self.fireSEBlock_2 = LargeKernelFireSEBlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.prm_2 = PRMLayer(groups=1)
        self.fireSEBlock_3 = LargeKernelFireSEBlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.prm_3 = PRMLayer(groups=1)
        self.fireSEBlock_4 = LargeKernelFireSEBlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireSEBlock_2(h2)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireSEBlock_3(h3)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.fireSEBlock_4(h4)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230413_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  use CASABlock_v1
    @thop.profile: macs=476404288.0, params=97378.0
    @model parameter number: 229758
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.cs_block_1 = CASABlock_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, head_num=8)
        self.pool_1 = nn.MaxPool2d((4, 4))
        self.cs_block_2 = CASABlock_v1(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, head_num=8)
        self.pool_2 = nn.MaxPool2d((4, 4))
        self.cs_block_3 = CASABlock_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, head_num=16)
        self.pool_3 = nn.MaxPool2d((4, 4), padding=(1,0))
        self.cs_block_4 = CASABlock_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, head_num=32)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding_1(h)
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h = self.cs_block_1(h)   # output shape: (batch_size, 32, 996, 128)
        h = self.pool_1(h)                  # output shape: (batch_size, 32, 249, 32)

        h = torch.cat((h, self.cs_block_2(h)), dim=1)   # output shape: (batch_size, 64, 249, 32)
        h = self.pool_2(h)                  # output shape: (batch_size, 64, 63, 8)

        h = torch.cat((h, self.cs_block_3(h)), dim=1)   # output shape: (batch_size, 128, 63, 8)
        h = self.pool_3(h)                  # output shape: (batch_size, 128, 16, 2)

        h = torch.cat((h, self.cs_block_4(h)), dim=1)   # output shape: (batch_size, 256, 16, 2)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230414_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  use LargeKernelFireSEBlock_v2
                  back to slice seq to 166*166
    @thop.profile: macs=791564416.0, params=731394.0
    @model parameter number: 903298
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = nn.Sequential(
            LargeKernelFireSEBlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1),
        )
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.fireSEBlock_2 = nn.Sequential(
            LargeKernelFireSEBlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1),
        )
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1,1))
        # self.prm_2 = PRMLayer(groups=1)
        self.fireSEBlock_3 = nn.Sequential(
            LargeKernelFireSEBlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1),
        )
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.fireSEBlock_4 = nn.Sequential(
            LargeKernelFireSEBlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1),
        )
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, h):
        h = self.embedding_1(h)
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 166)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 64, 83, 83)

        h = self.fireSEBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 128, 42, 42)

        h = self.fireSEBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.scale_conv_3(h)                  # output shape: (batch_size, 256, 21, 21)

        h = self.fireSEBlock_4(h)                 # output shape: (batch_size, 256, 21, 21)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230414_v2_copied_PretrainedModel_v20230413_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
                  replace all ReLU in PretrainedModel_v20230412_v3 with GELU
    @thop.profile: macs=2164168032.0, params=455954.0
    @model parameter number: 588007
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = LargeKernelFireSEBlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.fireSEBlock_2 = LargeKernelFireSEBlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.fireSEBlock_3 = LargeKernelFireSEBlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.fireSEBlock_4 = LargeKernelFireSEBlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        # h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireSEBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        # h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireSEBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        # h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.fireSEBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230415_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
                  replace all ReLU in PretrainedModel_v20230412_v3 with GELU
    @thop.profile: macs=2191201184.0, params=456914.0
    @model parameter number: 587994
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireSEBlock_1 = LargeKernelCABlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.fireSEBlock_2 = LargeKernelCABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.fireSEBlock_3 = LargeKernelCABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.fireSEBlock_4 = LargeKernelCABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireSEBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        # h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireSEBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        # h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireSEBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        # h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.fireSEBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230415_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
    @thop.profile: macs=2198807072.0, params=411594.0
    @model parameter number: 542702
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.sa_Block_1 = LargeKernelSABlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.sa_Block_2 = LargeKernelSABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.sa_Block_3 = LargeKernelSABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.sa_Block_4 = LargeKernelSABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.sa_Block_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        # h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.sa_Block_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        # h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.sa_Block_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        # h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.sa_Block_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230415_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireSEBlock_v1
                  replace all ReLU in PretrainedModel_v20230412_v3 with GELU
    @thop.profile: macs=3601585536.0, params=687674.0
    @model parameter number: 818790
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.ca_block_1 = LargeKernelCABlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8)
        self.sa_block_1 = LargeKernelSABlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.ca_block_2 = LargeKernelCABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.sa_block_2 = LargeKernelSABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.ca_block_3 = LargeKernelCABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.sa_block_3 = LargeKernelSABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.ca_block_4 = LargeKernelCABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.sa_block_4 = LargeKernelSABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.ca_block_1(h) # output shape: (batch_size, 32, 996, 128)
        h1 = self.sa_block_1(h1)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.ca_block_2(h1) # output shape: (batch_size, 64, 498, 64)
        h2 = self.sa_block_2(h2)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
                   
        h3 = self.ca_block_3(h2)    # output shape: (batch_size, 128, 249, 32)
        h3 = self.sa_block_3(h3)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.ca_block_4(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.sa_block_4(h4)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230415_v4(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
    @thop.profile: macs=2198807072.0, params=411594.0
    @model parameter number: 544606
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.sa_Block_1 = LargeKernelSABlock_v2(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, head_num=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.sa_Block_2 = LargeKernelSABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, head_num=32)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.sa_Block_3 = LargeKernelSABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, head_num=32)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.sa_Block_4 = LargeKernelSABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, head_num=32)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.sa_Block_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        # h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.sa_Block_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        # h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.sa_Block_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        # h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.sa_Block_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230416_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v4
    @thop.profile: macs=2191155264.0, params=411594.0
    @model parameter number: 542666
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_1 = PRMLayer(groups=1)
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        # self.prm_2 = PRMLayer(groups=1)
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        # self.prm_3 = PRMLayer(groups=1)
        self.fireBlock_4 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        # h2 = self.prm_1(h1)                         # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        # h3 = self.prm_2(h2)                         # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        # h4 = self.prm_3(h3)                         # output shape: (batch_size, 256, 125, 16)
        h4 = self.fireBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230416_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v5 to test the performance without Large Kernel
    @thop.profile: macs=2844549184.0, params=520394.0
    @model parameter number: 651466
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v5(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_2 = LargeKernelFireBlock_v5(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_3 = LargeKernelFireBlock_v5(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=96, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.fireBlock_4 = LargeKernelFireBlock_v5(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=192, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.fireBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230416_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v5 + PRM to test the performance without Large Kernel
    @thop.profile: macs=1872110112.0, params=356954.0
    @model parameter number: 488062
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelSABlock_v3(input_channels=32, squeeze_channels=8, e_1_channels=16, e_3_channels=16)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_2 = LargeKernelSABlock_v3(input_channels=64, squeeze_channels=16, e_1_channels=32, e_3_channels=32)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_3 = LargeKernelSABlock_v3(input_channels=128, squeeze_channels=32, e_1_channels=64, e_3_channels=64)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1,0))
        self.fireBlock_4 = LargeKernelSABlock_v3(input_channels=256, squeeze_channels=64, e_1_channels=128, e_3_channels=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 125, 16)

        h4 = self.fireBlock_4(h3)                 # output shape: (batch_size, 256, 125, 16)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230417_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v5 to test the performance without Large Kernel
    @thop.profile: macs=2072449088.0, params=129802.0
    @model parameter number: 260874
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v5(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_2 = LargeKernelFireBlock_v5(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.fireBlock_3 = LargeKernelFireBlock_v5(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)

        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v4
                  remove 256 channels layer in PretrainedModel_v20230416_v1
    @tthop.profile: macs=1582895168.0, params=102922.0
    @model parameter number: 233994
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v6
    @thop.profile: macs=2953646144.0, params=178186.0
    @model parameter number: 309258
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v6(input_channels=32, squeeze_channels=8, e_1_channels=8, e_5_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v6(input_channels=64, squeeze_channels=16, e_1_channels=16, e_5_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v6(input_channels=128, squeeze_channels=32, e_1_channels=32, e_5_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v7
    @thop.profile: macs=2953646144.0, params=178186.0
    @model parameter number: 309258
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v7(input_channels=32, squeeze_channels=8, e_1_channels=8, e_5_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v7(input_channels=64, squeeze_channels=16, e_1_channels=16, e_5_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v7(input_channels=128, squeeze_channels=32, e_1_channels=32, e_5_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v4(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v8
    @thop.profile: macs=2072449088.0, params=129802.0
    @model parameter number: 260874
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v8(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v8(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v8(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v5(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v9
    @thop.profile: macs=2072449088.0, params=129802.0
    @model parameter number: 260874
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v9(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v9(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v9(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v6(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v10
    @thop.profile: macs=2953646144.0, params=178186.0
    @model parameter number: 309258
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v10(input_channels=32, squeeze_channels=8, e_1_channels=8, e_5_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v10(input_channels=64, squeeze_channels=16, e_1_channels=16, e_5_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v10(input_channels=128, squeeze_channels=32, e_1_channels=32, e_5_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230418_v7(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v11
    @thop.profile: macs=3834843200.0, params=226570.0
    @model parameter number: 357642
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v11(input_channels=32, squeeze_channels=8, e_1_channels=8, e_7_channels=24, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v11(input_channels=64, squeeze_channels=16, e_1_channels=16, e_7_channels=48, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v11(input_channels=128, squeeze_channels=32, e_1_channels=32, e_7_channels=96, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230419_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12
    @thop.profile: macs=1484984384.0, params=97546.0
    @model parameter number: 228618
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230419_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v13
    @thop.profile: macs=1925582912.0, params=121850.0
    @model parameter number: 252922
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v13(input_channels=32, squeeze_channels=8, e_31_channels=16, e_33_channels=16, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v13(input_channels=64, squeeze_channels=16, e_31_channels=32, e_33_channels=32, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v13(input_channels=128, squeeze_channels=32, e_31_channels=64, e_33_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230419_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12 + PRMLayer
    @thop.profile: macs=1492123936.0, params=97546.0
    @model parameter number: 228639
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.prm_3 = PRMLayer(groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.prm_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230419_v4(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12 + PRMLayer(multi head)
    @thop.profile: macs=1492123936.0, params=97546.0
    @model parameter number: 228639
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=2)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.prm_3 = PRMLayer(groups=4)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.prm_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230421_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v13 + PRMLayer(multi head)
    @thop.profile: macs=1932722464.0, params=121850.0
    @model parameter number: 252987
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v13(input_channels=32, squeeze_channels=8, e_31_channels=16, e_33_channels=16, groups=1)
        self.prm_1 = PRMLayer(groups=2)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v13(input_channels=64, squeeze_channels=16, e_31_channels=32, e_33_channels=32, groups=1)
        self.prm_2 = PRMLayer(groups=4)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v13(input_channels=128, squeeze_channels=32, e_31_channels=64, e_33_channels=64, groups=1)
        self.prm_3 = PRMLayer(groups=8)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.prm_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230421_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12 + PRMLayer(multi head)
    @thop.profile: macs=1492123936.0, params=97546.0
    @model parameter number: 228639
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=2)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=4)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.cca_3 = CrissCrossAttention(in_dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.cca_3(h3)
        h3 = self.cca_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230421_v3(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12 + PRMLayer(multi head)
    @thop.profile: macs=2287808288.0, params=147082.0
    @model parameter number: 278427
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 128
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.cca_3 = CrissCrossAttention_v2(in_dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        h = h.reshape([-1, 1, h.shape[1], h.shape[2]]) # output shape: (batch_size, 1, 996, 128)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 996, 128)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 996, 128)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 498, 64)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 498, 64)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 249, 32)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 249, 32)
        h3 = self.cca_3(h3)
        h3 = self.cca_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230422_v1(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12
    @thop.profile: macs=332808384.0, params=97866.0
    @model parameter number: 267850
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)

        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230422_v2(nn.Module):
    """
    @description: model for 996bp, no cut, and use n*1 & 1*n conv (Asymmetric Convolution)
                  substitute pooling with conv
                  use LargeKernelFireBlock_v12
    @thop.profile: macs=461256288.0, params=390154.0
    @model parameter number: 560203
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=4)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=8)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.prm_3 = PRMLayer(groups=16)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_4 = LargeKernelFireBlock_v12(input_channels=256, squeeze_channels=64, e_1_channels=64, e_31_channels=128, e_33_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h3 = self.prm_3(h3)
        h3 = self.scale_conv_3(h3)                  # output shape: (batch_size, 256, 21, 21)
        h4 = self.fireBlock_4(h3)                 # output shape: (batch_size, 256, 21, 21)
        h = self.out_layer(h4)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230424_v1(nn.Module):
    """
    @description: use LargeKernelFireBlock_v12加non local的cca试一下
    @thop.profile: macs=470763281.0, params=125026.0
    @model parameter number: 295013
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.cca_1 = CrissCrossAttention(in_dim=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.cca_2 = CrissCrossAttention(in_dim=64)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.cca_3 = CrissCrossAttention(in_dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.cca_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.cca_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h3 = self.cca_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230424_v2(nn.Module):
    """
    @description: use LargeKernelFireBlock_v12加non local的cca_v2试一下
    @thop.profile: macs=620101393.0, params=163050.0
    @model parameter number: 333491
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v12(input_channels=32, squeeze_channels=8, e_1_channels=8, e_31_channels=16, e_33_channels=8, groups=1)
        self.cca_1 = CrissCrossAttention_v2(in_dim=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v12(input_channels=64, squeeze_channels=16, e_1_channels=16, e_31_channels=32, e_33_channels=16, groups=1)
        self.cca_2 = CrissCrossAttention_v2(in_dim=64)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v12(input_channels=128, squeeze_channels=32, e_1_channels=32, e_31_channels=64, e_33_channels=32, groups=1)
        self.cca_3 = CrissCrossAttention_v2(in_dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.cca_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.cca_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h3 = self.cca_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230425_v1(nn.Module):
    """
    @description: use LargeKernelFireBlock_v12加non local的cca_v2试一下
    @thop.profile: macs=641435409.0, params=168426.0
    @model parameter number: 338867
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.cca_1 = CrissCrossAttention_v2(in_dim=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.cca_2 = CrissCrossAttention_v2(in_dim=64)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.cca_3 = CrissCrossAttention_v2(in_dim=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.cca_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.cca_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h3 = self.cca_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230426_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v12加 2 * scaleconv试一下
    @thop.profile: macs=354142400.0, params=103242.0
    @model parameter number: 273226
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230426_v2(nn.Module):
    """
    @description: use LargeKernelFireBlock_v4 加PRM试一下
    @thop.profile: macs=355691104.0, params=103242.0
    @model parameter number: 273247
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=64)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.prm_3 = PRMLayer(groups=128)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.prm_1(h1)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.prm_2(h2)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h3 = self.prm_3(h3)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230430_v1(nn.Module):
    """
    @description: use 4 * LargeKernelFireBlock_v4 加 PRM 试一下
    @thop.profile: macs=355691104.0, params=103242.0
    @model parameter number: 273247
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.prm_1 = PRMLayer(groups=32)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.prm_2 = PRMLayer(groups=64)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.prm_3 = PRMLayer(groups=128)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_4 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.prm_4 = PRMLayer(groups=16)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.prm_1(h)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 64, 83, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.prm_2(h)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 128, 42, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.prm_3(h)
        h = self.scale_conv_3(h)                  # output shape: (batch_size, 256, 21, 21)
        h = self.fireBlock_4(h)
        h = self.prm_4(h)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230504_v1(nn.Module):
    """
    @description: use 4 * LargeKernelFireBlock_v4 + 3 * scaleconv
    @thop.profile: macs=488266944.0, params=411914.0
    @model parameter number: 581898
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_4 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 64, 83, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 128, 42, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.scale_conv_3(h)                  # output shape: (batch_size, 256, 21, 21)
        h = self.fireBlock_4(h)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230504_v2(nn.Module):
    """
    @description: use 3 * LargeKernelFireBlock_v4 + 2maxpool + 2trans1*1 试一下
    @thop.profile: macs=268464320.0, params=72522.0
    @model parameter number: 242506
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1))
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.scale_1(h)                  # output shape: (batch_size, 64, 83, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.scale_2(h)                  # output shape: (batch_size, 128, 42, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230504_v3(nn.Module):
    """
    @description: use 3 * LargeKernelFireBlock_v4 + 2maxpool 试一下
    @thop.profile: macs=231431744.0, params=59338.0
    @model parameter number: 229322
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,1))
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.scale_1(h)                  # output shape: (batch_size, 64, 83, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.scale_2(h)                  # output shape: (batch_size, 128, 42, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230505_v1(nn.Module):
    """
    @description: use 3 * LargeKernelFireBlock_v4 + 2maxpool 试一下
    @thop.profile: macs=869262528.0, params=225394.0
    @model parameter number: 395378
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(1,1))
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h = self.scale_1(h)                  # output shape: (batch_size, 64, 83, 83)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 64, 83, 83)
        h = self.scale_2(h)                  # output shape: (batch_size, 128, 42, 42)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230505_v2(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv试一下
    @thop.profile: macs=1384773952.0, params=401202.0
    @model parameter number: 571186
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))

        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230505_v3(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_new试一下
    @thop.profile: macs=1270536512.0, params=360434.0
    @model parameter number: 530418
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230505_v4(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
    @thop.profile: macs=1099180352.0, params=298898.0
    @model parameter number: 468882
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
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
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230505_v5(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v14 加 2 * scaleconv_new试一下
    @thop.profile: macs=1910556992.0, params=521938.0
    @model parameter number: 691922
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v14(input_channels=64, squeeze_channels=16, e_3_channels=32, e_5_channels=32, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v14(input_channels=128, squeeze_channels=32, e_3_channels=64, e_5_channels=64, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v14(input_channels=256, squeeze_channels=64, e_3_channels=128, e_5_channels=128, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230506_v1(nn.Module):
    """
    @description: use  3* LargeKernelSABlock_v2 加 2 * scaleconv_new试一下
    @thop.profile: macs=1273633920.0, params=360434.0
    @model parameter number: 530445
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 166
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelSABlock_v2(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, head_num=64, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelSABlock_v2(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, head_num=128, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelSABlock_v2(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, head_num=256, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 128)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 32, 166, 166)

        h1 = self.fireBlock_1(h)                  # output shape: (batch_size, 32, 166, 166)
        h1 = self.scale_conv_1(h1)                  # output shape: (batch_size, 64, 83, 83)
        h2 = self.fireBlock_2(h1)                 # output shape: (batch_size, 64, 83, 83)
        h2 = self.scale_conv_2(h2)                  # output shape: (batch_size, 128, 42, 42)
        h3 = self.fireBlock_3(h2)                 # output shape: (batch_size, 128, 42, 42)
        h = self.out_layer(h3)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230507_v1(nn.Module):
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
            nn.Linear(32, 2),
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

class PretrainedModel_v20230507_v2(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=29487168.0, params=77690.0
    @model parameter number: 94074
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=32, squeeze_channels=8, e_1_channels=8, e_3_channels=16, e_5_channels=8, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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

class PretrainedModel_v20230507_v3(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=69209152.0, params=120426.0
    @model parameter number: 136810
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
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=96, squeeze_channels=24, e_1_channels=24, e_3_channels=48, e_5_channels=24, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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

class PretrainedModel_v20230507_v4(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=133316672.0, params=360434.0
    @model parameter number: 376818
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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

class PretrainedModel_v20230508_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 + 3*prm 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=115144704.0, params=298898.0
    @model parameter number: 315309
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelSABlock_v2(input_channels=64,
                                                 squeeze_channels=16,
                                                 e_1_channels=16,
                                                 e_3_channels=32,
                                                 e_5_channels=16,
                                                 head_num=16,
                                                 groups=1)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelSABlock_v2(input_channels=128,
                                                 squeeze_channels=32,
                                                 e_1_channels=32,
                                                 e_3_channels=64,
                                                 e_5_channels=32,
                                                 head_num=32,
                                                 groups=1)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelSABlock_v2(input_channels=256,
                                                 squeeze_channels=64,
                                                 e_1_channels=64,
                                                 e_3_channels=128,
                                                 e_5_channels=64,
                                                 head_num=64,
                                                 groups=1)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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

class PretrainedModel_v20230510_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 + 3*CrissCrossAttention加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=160893688.0, params=406978.0
    @model parameter number: 423365
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.cca_1 = CrissCrossAttention(in_dim=64)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.cca_2 = CrissCrossAttention(in_dim=128)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.cca_3 = CrissCrossAttention(in_dim=256)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)
        
        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 166, 16)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166, 16)
        h = self.cca_1(h)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83, 8)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83, 8)
        h = self.cca_2(h)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42, 4)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42, 4)
        h = self.cca_3(h)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230510_v2(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 + 3*CrissCrossAttention_v2 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=223054584.0, params=558290.0
    @model parameter number: 575579
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.first_conv = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=(1, 1))
        self.fireBlock_1 = LargeKernelFireBlock_v4(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16, groups=1)
        self.cca_1 = CrissCrossAttention_v2(in_dim=64)
        self.scale_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1)),
        )
        self.fireBlock_2 = LargeKernelFireBlock_v4(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32, groups=1)
        self.cca_2 = CrissCrossAttention_v2(in_dim=128)
        self.scale_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1)),
        )
        self.fireBlock_3 = LargeKernelFireBlock_v4(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64, groups=1)
        self.cca_3 = CrissCrossAttention_v2(in_dim=256)
        self.out_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 166, 16)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166, 16)
        h = self.cca_1(h)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83, 8)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83, 8)
        h = self.cca_2(h)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42, 4)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42, 4)
        h = self.cca_3(h)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class PretrainedModel_v20230510_v3(nn.Module):
    """
    @description: use 3* LargeKernelFireBlock_v4 + 3*CrissCrossAttention_v2(sequeeze_scale=4) 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=169773816.0, params=428594.0
    @model parameter number: 445883
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
            nn.Linear(32, 2),
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
        h = self.pos_encoding(h)                    # output shape: (batch_size, 996, 16)

        slice_list = list()
        for i in range(11):
            slice_list.append(h[:, i*83:i*83+166, :])

        h = torch.stack(slice_list, dim=1)

        h = self.first_conv(h)                      # output shape: (batch_size, 64, 166, 16)

        h = self.fireBlock_1(h)                  # output shape: (batch_size, 64, 166, 16)
        h = self.cca_1(h)
        h = self.scale_conv_1(h)                  # output shape: (batch_size, 128, 83, 8)
        h = self.fireBlock_2(h)                 # output shape: (batch_size, 128, 83, 8)
        h = self.cca_2(h)
        h = self.scale_conv_2(h)                  # output shape: (batch_size, 256, 42, 4)
        h = self.fireBlock_3(h)                 # output shape: (batch_size, 256, 42, 4)
        h = self.cca_3(h)
        h = self.out_layer(h)                      # output shape: (batch_size, 2)
        return h

class AnchorModel_v20230515_v1(nn.Module):
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
            nn.Linear(32, 1),
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

class AnchorModel_v20230515_v2(nn.Module):
    """
    @description: use conv1d, slice_4*512 + 3* LargeKernelFireBlock1D_v1 加 2 * SliceLayer1D试一下
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
            nn.Linear(32, 1),
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

class AnchorModel_v20230516_v1(nn.Module):
    """
    @description: use  3* LargeKernelFireBlock_v4 加 2 * scaleconv_squeeze试一下
                  use 16 embedding dim
    @thop.profile: macs=114835488.0, params=298865.0
    @model parameter number: 315249
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
            nn.Linear(32, 1),
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

if __name__=="__main__":
    model = AnchorModel_v20230516_v1().cuda()

    from thop import profile
    input = torch.randint(4**5, (1, 996)).cuda()
    macs, params = profile(model, inputs=(input, )) # type: ignore
    print(f"thop.profile: macs={macs}, params={params}")

    # from torchstat import stat
    # stat(model, (996))

    print(f"model parameter number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

