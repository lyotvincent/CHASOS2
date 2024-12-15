'''
data generate & train process:
1. dnase_preprocessor.py gen_extension2_ocr
2. ocr_data_loader.py gen_OCR_ext2_data_by_chr
3. ocr_data_loader.py load_OCR_ext2_data_by_chr
4. ocr_trainer.py
'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from parameters import DNASE_DIR, CHROMOSOME_LIST, CELL_LINE_LIST
from pretrained_model.pytorch_glove.tools import KMerDictionary
from torch.utils.data import Dataset
from collections import Counter

def gen_OCR_data_by_chr(cell_line_list):
    kmd_5 = KMerDictionary(kmer_len=5)
    for cell_line in cell_line_list:
        print("processing cell_line:", cell_line)
        train_data_list = dict()
        train_label_list = dict()
        for chr in CHROMOSOME_LIST:
            train_data_list[chr] = list()
            train_label_list[chr] = list()
        with open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_ext.bed", "r") as f:
            for line in f:
                fields = line.strip().split()
                seq = fields[5]
                if "N" in seq: continue
                kmers = kmd_5.gen_kmers(seq, stride=1)
                train_data_list[fields[0]].append(kmers)
                train_label_list[fields[0]].append(float(fields[3]))
        #? the signalvalue in label should be make in a balanced ratio，restrict each chr's value distribution in max num of signalvalue=4
        #? 之前是把每个染色体上的所有信号值的数量上限限制为信号值4-5的数量，现在发现预测值分布和真实值不一样，还是不限制了。
        # for chr in CHROMOSOME_LIST:
        #     c = Counter()
        #     signalvalue4_num = 0
        #     for i in range(len(train_label_list[chr])):
        #         if 4 <= train_label_list[chr][i] < 5:
        #             signalvalue4_num += 1
        #     new_chr_train_data_list = list()
        #     new_chr_train_label_list = list()
        #     for i in range(len(train_label_list[chr])):
        #         if c[int(train_label_list[chr][i])] < signalvalue4_num:
        #             new_chr_train_data_list.append(train_data_list[chr][i])
        #             new_chr_train_label_list.append(train_label_list[chr][i])
        #             c[int(train_label_list[chr][i])] += 1
        #     train_data_list[chr] = new_chr_train_data_list
        #     train_label_list[chr] = new_chr_train_label_list
        for chr in CHROMOSOME_LIST:
            train_data_list[chr] = np.array(train_data_list[chr])
            print(f"train_data_list[{chr}].shape:", train_data_list[chr].shape)
            train_label_list[chr] = np.array(train_label_list[chr])
            print(f"train_label_list[{chr}].shape:", train_label_list[chr].shape)
            np.save(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_data.npy", train_data_list[chr])
            np.save(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_label.npy", train_label_list[chr])

def gen_OCR_ext2_data_by_chr(cell_line_list):
    kmd_5 = KMerDictionary(kmer_len=5)
    for cell_line in cell_line_list:
        print("processing cell_line:", cell_line)
        train_data_list = dict()
        train_label_list = dict()
        for chr in CHROMOSOME_LIST:
            train_data_list[chr] = list()
            train_label_list[chr] = list()
        with open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_ext2.bed", "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                seq = fields[5]
                if "N" in seq: continue
                kmers = kmd_5.gen_kmers(seq, stride=1)
                train_data_list[fields[0]].append(kmers)
                labels = fields[3].split(',')
                labels = [float(label) for label in labels]
                train_label_list[fields[0]].append(labels)
        for chr in CHROMOSOME_LIST:
            train_data_list[chr] = np.array(train_data_list[chr])
            print(f"train_data_list[{chr}].shape:", train_data_list[chr].shape)
            train_label_list[chr] = np.array(train_label_list[chr])
            print(f"train_label_list[{chr}].shape:", train_label_list[chr].shape)
            np.save(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_data.npy", train_data_list[chr])
            np.save(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_label.npy", train_label_list[chr])

def load_OCR_data_by_chr(cell_line_list, val_chr_list):
    train_data_list = np.array([])
    train_label_list = np.array([])
    val_data_list = np.array([])
    val_label_list = np.array([])
    for cell_line in cell_line_list:
        for chr in CHROMOSOME_LIST:
            current_data_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_data.npy")
            current_label_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_label.npy")
            if len(current_data_list) == 0: continue
            if chr in val_chr_list:
                if len(val_data_list) == 0:
                    val_data_list = current_data_list
                    val_label_list = current_label_list
                else:
                    val_data_list = np.concatenate((val_data_list, current_data_list), axis=0)
                    val_label_list = np.concatenate((val_label_list, current_label_list), axis=0)
            else:
                if len(train_data_list) == 0:
                    train_data_list = current_data_list
                    train_label_list = current_label_list
                else:
                    train_data_list = np.concatenate((train_data_list, current_data_list), axis=0)
                    train_label_list = np.concatenate((train_label_list, current_label_list), axis=0)
    
    return train_data_list, train_label_list, val_data_list, val_label_list

def load_OCR_data_by_cell_line(cell_line_list, val_cell_line_list):
    train_data_list = np.array([])
    train_label_list = np.array([])
    val_data_list = np.array([])
    val_label_list = np.array([])
    for cell_line in cell_line_list:
        if cell_line in val_cell_line_list:
            for chr in CHROMOSOME_LIST:
                current_data_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_data.npy")
                current_label_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_label.npy")
                if len(current_data_list) == 0: continue
                if len(val_data_list) == 0:
                    val_data_list = current_data_list
                    val_label_list = current_label_list
                else:
                    val_data_list = np.concatenate((val_data_list, current_data_list), axis=0)
                    val_label_list = np.concatenate((val_label_list, current_label_list), axis=0)
        else:
            for chr in CHROMOSOME_LIST:
                current_data_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_data.npy")
                current_label_list = np.load(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_{chr}_label.npy")
                if len(current_data_list) == 0: continue
                if len(train_data_list) == 0:
                    train_data_list = current_data_list
                    train_label_list = current_label_list
                else:
                    train_data_list = np.concatenate((train_data_list, current_data_list), axis=0)
                    train_label_list = np.concatenate((train_label_list, current_label_list), axis=0)
    
    return train_data_list, train_label_list, val_data_list, val_label_list



class OCRDataset(Dataset):
    def __init__(self, kmer_data, labels, is_cuda=False):
        self.kmer_data = torch.tensor(kmer_data)
        self.labels = torch.tensor(labels, dtype=torch.float)
        if is_cuda:
            self.kmer_data = self.kmer_data.cuda()
            self.labels = self.labels.cuda()

    def __len__(self):
        return len(self.kmer_data)

    def __getitem__(self, index):
        return self.kmer_data[index], self.labels[index]


if __name__ == "__main__":
    # gen_OCR_data_by_chr(["GM12878"])
    # gen_OCR_ext2_data_by_chr(["GM12878", "HCT116", "IMR-90", "K562"])
    gen_OCR_ext2_data_by_chr(["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"])
    # load_OCR_data_by_chr(["GM12878"], ["chr22", "chrX"])

