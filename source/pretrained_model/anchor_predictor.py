


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained_model'))

import random
import numpy as np
from parameters import PRETRAINED_MODEL_PATH
from pretrained_model.models import *
from pretrained_model.pytorch_glove.tools import KMerDictionary

def seed_torch(seed=9523):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU. torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False # type: ignore 
    torch.backends.cudnn.deterministic = True  # 选择确定性算法 # type: ignore

seed_torch()

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # torch.backends.cudnn.enabled = True # type: ignore
        # torch.backends.cudnn.benchmark = True # type: ignore
    else:
        device = torch.device('cpu')
    return device

def get_anchor_model(device, model_path=None):
    model = AnchorModel_v20230516_v1()
    if model_path is not None:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt')
    saved_dict = checkpoint['model']
    model.load_state_dict(saved_dict)
    model.to(device)
    model.eval()
    return model

def init_predictor(model_path=None):
    device = get_device()
    model = get_anchor_model(device, model_path)
    kmd_5 = KMerDictionary(kmer_len=5)
    return model, device, kmd_5

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

def get_pred_scores(model, test_kmer_data, device):
    test_kmer_data = test_kmer_data.to(device)
    # print(test_kmer_data.shape)
    test_pred_scores = model(test_kmer_data)
    return test_pred_scores

if __name__ == "__main__":
    model, device, kmd = init_predictor()
    seq = 'CTAAAAATACAAAATTAGCCGGTCGTGGCGGCGCGCGCCTGTAATCCTAATTACTCGGGAGGCTGAGACAGGAGAATTACTTGAACCCAGGAGGCAGAGGTTGCAATGAGCCGAGATTGCACCACTGCACTCCAGCCTGGGCAACAAGCGCGAAACTCTGTCGCAAAAAATAAATAAATACATACATACATACATACAAATATTGCCAATTTGATAGGCAAAAATATATGCAAATTATGTTTTAACAAATATTAATAAAGTTGAACAATGGCCCAATGTAATTTGTTACCTATCTGAAAGACATGCACATTCAGGGAGTTACAGTACTTTTTCATGTCTTGCAGCATCAACAAATTGTTTTCTAAATGTGGAAAAAGAGACATTCAGTACAGGCTAACGTAACGTGGGACAGCATGCAAAGCCATGAGAAGTGGGAAAATGAGAGAGAATCTGGGCGTGGCTTACAGGGCTGCTGAAGTAAGACACACCCTTCGCTTCACCTGTGGCCTCCCGCGGCGGCTGCGGCTCCTTCGCAGGACCGGTAGGGGGCGCGCGCGGTTGAGTCCAGCAATTGCGAGGACCTCCGCAGGCGCAGCCCAGCACTGACGCCTCTCCGGGCCGTGGCTCCTCTTTCCCAGGCCGAAAGCGGCCGGGCCCTCTGCTCCGCTGCGCCCTGGCGCGGCCGCACCCACTGCGCGTTTGTGAGCTGGCCATCGGTCACACTGGGATACTCGAGGAGGAGGCCTGGGGCCTGCATTCAATTCCCTAAGAGGGACCCGCGCTGGCCGCTGCAGGCGGGTGGAGAACGGGCGTCTGAGTCTTTGGCTTTCGTACCGGAATCTCCCCGGGAGATTTTACAAGCCCCTCCGAATGTTCTGATTGAATTGGTCTGGGGGGGTCGGTCCGGGCATCTAACGAGCAGCCGGGATGGAGAATTACTAGTCTGGATTCTGGTCCTCAGAGGTTCACTCACGTTAATCAACCCGAGTCCATCTTGGTG'
    kmers = encode_kmer(kmd, [seq])
    print(kmers.shape)
    test_pred_scores = get_pred_scores(model, kmers, device)
    print(test_pred_scores)
    print(test_pred_scores[0])
    print(test_pred_scores[0][0].item())

