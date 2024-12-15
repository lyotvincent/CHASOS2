

from custom_layer import *
from positional_encoding import PositionalEncoding
from models import AnchorModel_v20230516_v1
import torch, math, random, time
import torch.nn as nn
from parameters import PRETRAINED_MODEL_PATH
from pretrained_data_loader import load_kmer_encoded_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_glove.tools import KMerDictionary

start_time = time.time()

model = AnchorModel_v20230516_v1()
checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_06_29_20_59_01.pt')
saved_dict = checkpoint['model']
print(len(saved_dict))
print(len(model.state_dict()))
model.load_state_dict(saved_dict)
model.cuda()
model.eval()

# SPLIT_MODE = 'chr'
# cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
# VAL_CHR_LIST = ['chr22', 'chrX']
# STRIDE = 1
# _, _, kmer_val_data_list, labels = load_kmer_encoded_data(cell_line_list=cell_line_list, split_mode=SPLIT_MODE, val_chr_list=VAL_CHR_LIST, stride=STRIDE)
# labels = labels[:, 0]

base_dict = {0:"A",1:"T",2:"C",3:"G"}

seq_pattern_A = "A"*483+"{}"+"A"*483
seq_pattern_T = "T"*483+"{}"+"T"*483
seq_pattern_C = "C"*483+"{}"+"C"*483
seq_pattern_G = "G"*483+"{}"+"G"*483

def gen_sim_CTCF_motif():
    sim_CTCF_motif = ""
    for i in range(34):
        sim_CTCF_motif += base_dict[random.randint(0, 3)]
    return sim_CTCF_motif

motifs = list()
while len(motifs) < 100000:
    if len(motifs) % 1000 == 0:
        print("gen",len(motifs))
    sim_CTCF_motif = gen_sim_CTCF_motif()
    if sim_CTCF_motif not in motifs:
        motifs.append(sim_CTCF_motif)


kmd_5 = KMerDictionary(kmer_len=5)

def encode_kmer(kmd, seqs, stride=1):
    '''
    kmd : KMerDictionary
    '''
    kmer_seqs = list()
    for seq in seqs:
        kmers = kmd.gen_kmers(seq, stride=stride)
        kmer_seqs.append(kmers)
    kmer_seqs = torch.tensor(kmer_seqs, dtype=torch.int)
    return kmer_seqs

CTCF_motif_score = dict()
CTCF_motif_score["A"] = [0.]*34
CTCF_motif_score["T"] = [0.]*34
CTCF_motif_score["C"] = [0.]*34
CTCF_motif_score["G"] = [0.]*34

def gen_seq(seq_pattern, motif):
    seq = seq_pattern.format(motif)
    seq = encode_kmer(kmd_5, [seq])
    seq = seq.cuda()
    return seq

def gen_preds(seq):
    preds = model(seq)
    preds = preds.cpu()
    preds = preds[0][0].item()
    return preds

with torch.no_grad():
    for j, motif in enumerate(motifs):
        if j % 1000 == 0:
            print("pred",j)
        seq = gen_seq(seq_pattern_A, motif)
        preds = gen_preds(seq)
        for i, s in enumerate(motif):
            CTCF_motif_score[s][i] += preds
        # print(CTCF_motif_score)
        seq = gen_seq(seq_pattern_T, motif)
        preds = gen_preds(seq)
        for i, s in enumerate(motif):
            CTCF_motif_score[s][i] += preds
        seq = gen_seq(seq_pattern_C, motif)
        preds = gen_preds(seq)
        for i, s in enumerate(motif):
            CTCF_motif_score[s][i] += preds
        seq = gen_seq(seq_pattern_G, motif)
        preds = gen_preds(seq)
        for i, s in enumerate(motif):
            CTCF_motif_score[s][i] += preds

for i in range(34):
    s = CTCF_motif_score["A"][i]+CTCF_motif_score["T"][i]+CTCF_motif_score["C"][i]+CTCF_motif_score["G"][i]
    CTCF_motif_score["A"][i] /= s
    CTCF_motif_score["T"][i] /= s
    CTCF_motif_score["C"][i] /= s
    CTCF_motif_score["G"][i] /= s

# print(CTCF_motif_score["A"])
# print(CTCF_motif_score["T"])
# print(CTCF_motif_score["C"])
# print(CTCF_motif_score["G"])

print(f"time: {time.time()-start_time}s")

print(CTCF_motif_score["A"]+CTCF_motif_score["T"]+CTCF_motif_score["C"]+CTCF_motif_score["G"])

