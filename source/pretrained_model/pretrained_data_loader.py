'''
@author: 孙嘉良
@purpose: load pretrained data
'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from pytorch_glove.tools import KMerDictionary
from parameters import PRETRAINED_DATA_PATH, CELL_LINE_LIST, TRAIN_SET_RATIO, PRETRAINED_MODEL_PATH, CHROMOSOME_LIST
from collections import defaultdict
from utils import TimeStatistic

@TimeStatistic
def load_kmer_seqs(cell_line_list, split_mode, stride=1):
    '''
    @purpose: load data from csv file and split train/validation set
    @params: cell_line_list: list of cell line
    @params: split_mode: 'random', 'chr', 'cell_line'
    '''
    assert split_mode in ['random', 'chr', 'cell_line']
    random.seed(9523)
    kmer_train_data_dict = defaultdict(list)
    train_label_dict = defaultdict(list)
    kmer_val_data_dict = defaultdict(list)
    val_label_dict = defaultdict(list)
    monomer_train_data_dict = defaultdict(list)
    monomer_val_data_dict = defaultdict(list)
    kmd_5 = KMerDictionary(kmer_len=5)
    kmd_1 = KMerDictionary(kmer_len=1)
    VAL_CHR_LIST = ['chr6', 'chr9']
    VAL_CELL_LINE_LIST = ['GM12878', 'K562']

    def load_a_file_and_split_random(file_path, label):
        '''
        @purpose: split train/val randomly
        '''
        seqs_corpus = list()
        with open(file_path, 'r') as f:
            for line in f:
                seq = line.strip().split()[3]
                kmers = kmd_5.gen_kmers(seq, stride=stride)
                seqs_corpus.append(kmers)
        seqs_corpus = np.array(seqs_corpus)

        # split train/val
        indices = np.arange(len(seqs_corpus))
        np.random.shuffle(indices)
        train_size = int(len(indices) * TRAIN_SET_RATIO) # 80% train set
        test_size = len(indices) - train_size # 20% test set

        kmer_train_data_dict[cell_line].extend(seqs_corpus[indices[:train_size]])
        train_label_dict[cell_line].extend(label * train_size)
        kmer_val_data_dict[cell_line].extend(seqs_corpus[indices[train_size:]])
        val_label_dict[cell_line].extend(label * test_size)

    def load_and_split_by_chr_old(file_path, label):
        '''
        @purpose: split train/val by chromosome, but this function only can split solid data
        '''
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                seq = fields[3]
                kmers = kmd_5.gen_kmers(seq, stride=stride)
                monomers = kmd_1.gen_kmers(seq)
                if fields[0] in VAL_CHR_LIST:
                    kmer_val_data_dict[cell_line].append(kmers)
                    val_label_dict[cell_line].append(label[0])
                    monomer_val_data_dict[cell_line].append(monomers)
                else:
                    kmer_train_data_dict[cell_line].append(kmers)
                    train_label_dict[cell_line].append(label[0])
                    monomer_train_data_dict[cell_line].append(monomers)

    def load_and_split_by_chr(file_path, label):
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                seq = fields[3]
                kmers = kmd_5.gen_kmers(seq, stride=stride)
                kmer_train_data_dict[fields[0]].append(kmers)
                train_label_dict[fields[0]].append(label[0])

    def load_and_split_by_cell_line(file_path, label):
        '''
        @purpose: split train/val by cell line
        '''
        pass

    split_mode_dict = {'random':load_a_file_and_split_random, 'chr':load_and_split_by_chr, 'cell_line':load_and_split_by_cell_line}
    for i, cell_line in enumerate(cell_line_list):
        print(f"loading {i} {cell_line}...")
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ocr.csv", [[1., 1.]])
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ccr.csv", [[1., 0.]])
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ocr.csv", [[0., 1.]])
        split_mode_dict[split_mode](PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ccr.csv", [[0., 0.]])
    return kmer_train_data_dict, train_label_dict, kmer_val_data_dict, val_label_dict

class PretrainedDataset(Dataset):
    def __init__(self, kmer_data, monomer_data, labels):
        self.kmer_data = torch.tensor(kmer_data)
        if monomer_data:
            self.monomer_data = torch.tensor(monomer_data)
        else:
            self.monomer_data = None
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.kmer_data)

    def __getitem__(self, index):
        if self.monomer_data:
            return self.kmer_data[index], self.monomer_data[index], self.labels[index]
        else:
            return self.kmer_data[index], self.labels[index]

@TimeStatistic
def load_kmer_encoded_data(cell_line_list, split_mode, val_chr_list=['chr6', 'chr9'], stride=1):
    if split_mode == "random":
        # 默认每个cell line都随机82分
        pretrained_encoded_data_path = PRETRAINED_MODEL_PATH+f'/encoded_data/{split_mode}/stride_{stride}/'
        kmer_train_data_path = pretrained_encoded_data_path+r'/kmer_train_data_dict.pkl'
        kmer_val_data_path = pretrained_encoded_data_path+r'/kmer_val_data_dict.pkl'
        train_label_path = pretrained_encoded_data_path+r'/train_label_dict.pkl'
        val_label_path = pretrained_encoded_data_path+r'/val_label_dict.pkl'
        if os.path.exists(kmer_train_data_path) and\
        os.path.exists(kmer_val_data_path) and\
        os.path.exists(train_label_path) and\
        os.path.exists(val_label_path):
            with open(kmer_train_data_path, "rb") as f:
                kmer_train_data_dict = pickle.load(f)
            with open(kmer_val_data_path, "rb") as f:
                kmer_val_data_dict = pickle.load(f)
            with open(train_label_path, "rb") as f:
                train_label_dict = pickle.load(f)
            with open(val_label_path, "rb") as f:
                val_label_dict = pickle.load(f)
        else:
            kmer_train_data_dict, train_label_dict, kmer_val_data_dict, val_label_dict = load_kmer_seqs(CELL_LINE_LIST, split_mode=split_mode, stride=stride)
            with open(kmer_train_data_path, "wb") as f:
                pickle.dump(kmer_train_data_dict, f)
            with open(kmer_val_data_path, "wb") as f:
                pickle.dump(kmer_val_data_dict, f)
            with open(train_label_path, "wb") as f:
                pickle.dump(train_label_dict, f)
            with open(val_label_path, "wb") as f:
                pickle.dump(val_label_dict, f)
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = list(), list(), list(), list()
        for cell_line in CELL_LINE_LIST:
            kmer_train_data_list.extend(kmer_train_data_dict[cell_line])
            kmer_train_label_list.extend(train_label_dict[cell_line])
            kmer_val_data_list.extend(kmer_val_data_dict[cell_line])
            kmer_val_label_list.extend(val_label_dict[cell_line])
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array(kmer_train_data_list), np.array(kmer_train_label_list), np.array(kmer_val_data_list), np.array(kmer_val_label_list)
    elif split_mode == 'chr':
        pretrained_encoded_data_path = PRETRAINED_MODEL_PATH+f'/encoded_data/{split_mode}/stride_{stride}/'
        files_complete = True
        for cell_line in cell_line_list:
            for chr in CHROMOSOME_LIST:
                if os.path.exists(pretrained_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy') == False or\
                   os.path.exists(pretrained_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy') == False:
                    files_complete = False
                    break
            if files_complete == False:
                break
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array([]), np.array([]), np.array([]), np.array([])
        if files_complete:
            for cell_line in cell_line_list:
                for chr in CHROMOSOME_LIST:
                    feature_data = np.load(pretrained_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy')
                    labels = np.load(pretrained_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy')
                    print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
                    if len(feature_data) == 0: continue
                    if chr in val_chr_list:
                        if len(kmer_val_data_list) == 0:
                            kmer_val_data_list = feature_data
                        else:
                            kmer_val_data_list = np.concatenate((kmer_val_data_list, feature_data), axis=0)
                        if len(kmer_val_label_list) == 0:
                            kmer_val_label_list = labels
                        else:
                            kmer_val_label_list = np.concatenate((kmer_val_label_list, labels), axis=0)
                    else:
                        if len(kmer_train_data_list) == 0:
                            kmer_train_data_list = feature_data
                        else:
                            kmer_train_data_list = np.concatenate((kmer_train_data_list, feature_data), axis=0)
                        if len(kmer_train_label_list) == 0:
                            kmer_train_label_list = labels
                        else:
                            kmer_train_label_list = np.concatenate((kmer_train_label_list, labels), axis=0)
        else:
            for cell_line in cell_line_list:
                kmer_train_data_dict, train_label_dict, _, _ = load_kmer_seqs([cell_line], split_mode=split_mode, stride=stride)
                for chr in CHROMOSOME_LIST:
                    print(f'{cell_line}_data_list_{chr}.npy & {cell_line}_labels_list_{chr}.npy are saving...')
                    np.save(pretrained_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy', np.array(kmer_train_data_dict[chr]))
                    np.save(pretrained_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy', np.array(train_label_dict[chr]))
    else:
        kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array([]), np.array([]), np.array([]), np.array([])
    return kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list

def load_kmer_encoded_data_div_cell_line(cell_line_list, val_cell_line_list=['chr6', 'chr9']):
    pretrained_encoded_data_path = PRETRAINED_MODEL_PATH+f'/encoded_data/chr/stride_1/'
    kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = np.array([]), np.array([]), np.array([]), np.array([])
    for cell_line in cell_line_list:
        if cell_line in val_cell_line_list:
            for chr in CHROMOSOME_LIST:
                feature_data = np.load(pretrained_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy')
                labels = np.load(pretrained_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy')
                print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
                if len(feature_data) == 0: continue
                if len(kmer_val_data_list) == 0:
                    kmer_val_data_list = feature_data
                else:
                    kmer_val_data_list = np.concatenate((kmer_val_data_list, feature_data), axis=0)
                if len(kmer_val_label_list) == 0:
                    kmer_val_label_list = labels
                else:
                    kmer_val_label_list = np.concatenate((kmer_val_label_list, labels), axis=0)
        else:
            for chr in CHROMOSOME_LIST:
                feature_data = np.load(pretrained_encoded_data_path+f'/{cell_line}_data_list_{chr}.npy')
                labels = np.load(pretrained_encoded_data_path+f'/{cell_line}_labels_list_{chr}.npy')
                print(f"loading & concating {cell_line} {chr}, data len={len(feature_data)}")
                if len(feature_data) == 0: continue
                if len(kmer_train_data_list) == 0:
                    kmer_train_data_list = feature_data
                else:
                    kmer_train_data_list = np.concatenate((kmer_train_data_list, feature_data), axis=0)
                if len(kmer_train_label_list) == 0:
                    kmer_train_label_list = labels
                else:
                    kmer_train_label_list = np.concatenate((kmer_train_label_list, labels), axis=0)
    return kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list


if __name__=="__main__":
    ## 假设我们有一个数据集，包含了 100 个样本，每个样本是一个长度为 10 的序列，每个样本还带有一个标签。
    # data = [torch.randn(10) for i in range(100)]
    # labels = [torch.randint(0, 2, (1,)).item() for i in range(100)]
    # data = list(data)
    # labels = list(labels)
    # print(type(data), type(labels))
    ## 使用 PretrainedDataset 包装数据集
    # my_dataset = PretrainedDataset(data, labels)
    ## 创建 DataLoader，指定 batch_size 为 10
    # my_dataloader = DataLoader(my_dataset, batch_size=10, shuffle=True)
    ## 遍历 DataLoader
    # for batch in my_dataloader:
    #     batch_data, batch_labels = batch
    #     print(type(batch_data), type(batch_labels))
    #     print(batch_data.shape, batch_labels.shape)
    #     exit()



    ## 默认每个cell line都82分
    # train_data_dict, train_label_dict, val_data_dict, val_label_dict = load_kmer_seqs(CELL_LINE_LIST, split_mode='chr')
    # train_data_list, train_label_list, val_data_list, val_label_list = list(), list(), list(), list()
    # for cell_line in CELL_LINE_LIST:
    #     train_data_list.extend(train_data_dict[cell_line])
    #     train_label_list.extend(train_label_dict[cell_line])
    #     val_data_list.extend(val_data_dict[cell_line])
    #     val_label_list.extend(val_label_dict[cell_line])
    # print(len(train_data_list))
    # print(len(train_label_list))
    # print(len(val_data_list))
    # print(len(val_label_list))
    # # TODO leave one cell line out (LOCLO)

    # my_dataset = PretrainedDataset(train_data_list, train_label_list)
    # my_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True)
    # for batch in my_dataloader:
    #     batch_data, batch_labels = batch
    #     print(batch_data.shape, batch_labels)
    #     exit()

    # 用load_kmer_encoded_data()函数生成数据或加载数据
    kmer_train_data_list, kmer_train_label_list, kmer_val_data_list, kmer_val_label_list = load_kmer_encoded_data(cell_line_list=CELL_LINE_LIST, split_mode='chr', val_chr_list=['chr22', 'chrX'], stride=1)
    # print(len(kmer_train_data_list))
    # print(len(kmer_train_label_list))
    # print(len(kmer_val_data_list))
    # print(len(kmer_val_label_list))


