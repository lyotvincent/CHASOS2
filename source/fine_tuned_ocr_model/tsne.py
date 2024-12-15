

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained_model.custom_layer import *
from positional_encoding import PositionalEncoding
import torch, math
import torch.nn as nn
from parameters import FINE_TUNED_OCR_MODEL_PATH
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ocr_data_loader import load_OCR_data_by_chr

class tsne_OCRModel_v20230524_v1(nn.Module):
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


model = tsne_OCRModel_v20230524_v1()
checkpoint = torch.load(FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_06_30_23_34_50.pt')
saved_dict = checkpoint['model']
print(len(saved_dict))
print(len(model.state_dict()))
# saved_dict.pop("out_layer.4.weight")
# saved_dict.pop("out_layer.4.bias")
model.load_state_dict(saved_dict)
model.cuda()
model.eval()

SPLIT_MODE = 'chr'
cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
VAL_CHR_LIST = ['chr22', 'chrX']
STRIDE = 1
_, _, kmer_val_data_list, labels = load_OCR_data_by_chr(cell_line_list=cell_line_list, val_chr_list=VAL_CHR_LIST)
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
colors = list()
for l in labels:
    if l < 0.00001:
        colors.append("#ED1C24")
    elif 0.05 > l >= 0.00001:
        colors.append("#FF7F27")
    elif 0.1>l>=0.05:
        colors.append("#FFF200")
    elif 0.2>l>=0.1:
        colors.append("#22B14C")
    elif 0.3>l>=0.2:
        colors.append("#00A2E8")
    elif 0.4>l>=0.3:
        colors.append("#3F48CC")
    elif l>=0.4:
        colors.append("#A349A4")
    else:
        print(f"error, l={l}")
# 可视化结果
plt.figure(figsize=(6, 5))
plt.scatter(result[:, 0], result[:, 1], c=colors, s=5)  # s表示点的大小
plt.title('t-SNE visualization of 6645 data points')
plt.tight_layout()
plt.show()

