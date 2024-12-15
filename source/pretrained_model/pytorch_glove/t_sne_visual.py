import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
from parameters import GLOVE_PATH, CELL_LINE_LIST

def draw_t_sne(cell_line):
    '''
    @draw t-SNE for embedded cell line pretrained data
    '''
    embed = nn.Embedding(num_embeddings=4**5, embedding_dim=128).type(torch.float64)
    saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
    embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
    model_dict = embed.state_dict()
    model_dict.update(embedding_dict)
    embed.load_state_dict(model_dict)
    # embed.to("cuda")

    DOC_PATH = GLOVE_PATH+f'/data/pretrained_{cell_line}_corpus.pickle'
    with open(DOC_PATH, 'rb') as fp:
        doc = pickle.load(fp)
    seqs = [seq for seq, _ in doc]
    labels = [label for _, label in doc]
    seqs = torch.tensor(seqs)
    # seqs = seqs.to("cuda")
    print(seqs.shape)

    embed.eval()
    with torch.no_grad():
        seq_embeds = list()
        for seq in seqs:
            seq_embed = embed(seq).numpy()
            seq_embeds.append(seq_embed)
        print(len(seq_embeds))
    seq_embeds = np.array(seq_embeds)
    labels = np.array(labels)

    # 将数据降维到二维
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(seq_embeds.reshape((seq_embeds.shape[0], -1)))

    # 将结果可视化
    FONTSIZE = 8
    plt.figure(dpi=600, figsize=(8, 8))
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
    plt.rcParams.update(params)
    plt.figure()
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1], s=1, c='#ED1C24', label='pos_an_ocr')
    plt.scatter(tsne_results[labels==2, 0], tsne_results[labels==2, 1], s=1, c='#22B14C', label='pos_an_ccr')
    plt.scatter(tsne_results[labels==3, 0], tsne_results[labels==3, 1], s=1, c='#00A2E8', label='neg_an_ocr')
    plt.scatter(tsne_results[labels==4, 0], tsne_results[labels==4, 1], s=1, c='#FFC90E', label='neg_an_ccr')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GLOVE_PATH+f'/figures/{cell_line}.svg', format='svg', bbox_inches='tight')

def draw_tensorboard(cell_line):
    embed = nn.Embedding(num_embeddings=4**5, embedding_dim=128).type(torch.float64)
    saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
    embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
    model_dict = embed.state_dict()
    model_dict.update(embedding_dict)
    embed.load_state_dict(model_dict)

    DOC_PATH = GLOVE_PATH+f'/data/pretrained_{cell_line}_corpus.pickle'
    with open(DOC_PATH, 'rb') as fp:
        doc = pickle.load(fp)
    num = 25
    seqs = list()
    labels = list()
    counter1, counter2, counter3, counter4 = 0, 0, 0, 0
    for seq, label in doc:
        if label == 1 and counter1 < num:
            seqs.append(seq)
            labels.append(label)
            counter1 += 1
        elif label == 2 and counter2 < num:
            seqs.append(seq)
            labels.append(label)
            counter2 += 1
        elif label == 3 and counter3 < num:
            seqs.append(seq)
            labels.append(label)
            counter3 += 1
        elif label == 4 and counter4 < num:
            seqs.append(seq)
            labels.append(label)
            counter4 += 1
    # seqs = [seq for seq, _ in doc]
    # labels = [label for _, label in doc]
    seqs = torch.tensor(seqs)
    # seqs = seqs.to("cuda")
    print(seqs.shape)

    embed.eval()
    with torch.no_grad():
        seq_embeds = list()
        for seq in seqs:
            seq_embed = embed(seq).numpy()
            seq_embeds.append(seq_embed)
        print(len(seq_embeds))
    seq_embeds = np.array(seq_embeds)
    labels = np.array(labels)
    mat = seq_embeds.reshape((seq_embeds.shape[0], -1))#图像展开成一维向量
    writer = SummaryWriter(GLOVE_PATH+f'/figures/{cell_line}_embedding_tensorboard')
    writer.add_embedding(mat,metadata=labels)
    writer.close()

if __name__=='__main__':
    for cell_line in ['GM12878']:
        # draw_t_sne(cell_line)
        draw_tensorboard(cell_line)


