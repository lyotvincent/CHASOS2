

from custom_layer import *
from positional_encoding import PositionalEncoding
import torch, math
import torch.nn as nn
from parameters import PRETRAINED_MODEL_PATH
from pretrained_data_loader import load_kmer_encoded_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class tsne_AnchorModel_v20230516_v1(nn.Module):
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
            nn.ReLU()
        )


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
        h = self.out_layer(h3)                      # output shape: (batch_size, 32)
        return h

model = tsne_AnchorModel_v20230516_v1()
checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_06_29_20_59_01.pt')
saved_dict = checkpoint['model']
print(len(saved_dict))
print(len(model.state_dict()))
saved_dict.pop("out_layer.4.weight")
saved_dict.pop("out_layer.4.bias")
model.load_state_dict(saved_dict)
model.cuda()
model.eval()

SPLIT_MODE = 'chr'
cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
VAL_CHR_LIST = ['chr22', 'chrX']
STRIDE = 1
_, _, kmer_val_data_list, labels = load_kmer_encoded_data(cell_line_list=cell_line_list, split_mode=SPLIT_MODE, val_chr_list=VAL_CHR_LIST, stride=STRIDE)
labels = labels[:, 0]

BATCH_SIZE = 256
kmer_val_data_list = torch.tensor(kmer_val_data_list).cpu()
test_pred_scores = model(kmer_val_data_list[:BATCH_SIZE].cuda()).cpu()
with torch.no_grad():
    for i in range(1, math.ceil(len(kmer_val_data_list)/BATCH_SIZE)):
        print(i)
        tempval = kmer_val_data_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE].cuda()
        preds = model(tempval)
        preds = preds.cpu()
        test_pred_scores = torch.cat([test_pred_scores, preds], dim=0).cpu()
        print(test_pred_scores.shape)

# t-SNE the preds
# 将张量转换为numpy数组，因为TSNE是在numpy环境下运行的
tensor_np = test_pred_scores.detach().numpy()

# 使用TSNE进行降维
tsne = TSNE(n_components=2, random_state=0)
result = tsne.fit_transform(tensor_np)

# print(labels)
colors = ['red' if l == 0 else 'blue' for l in labels]
# 可视化结果
plt.figure(figsize=(6, 5))
plt.scatter(result[:, 0], result[:, 1], c=colors, s=5)  # s表示点的大小
plt.title('t-SNE visualization of 6645 data points')
plt.tight_layout()
plt.show()

