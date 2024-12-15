'''
@description: This script preprocesses the data for the Lollipop experiment.
              chrom
              start1
              start2
              response
              length
              motif_pattern
              avg_motif_strength
              std_motif_strength
              avg_conservation
              std_conservation
              avg_HS
              std_HS
              HS_in-between
              HS_left
              HS_right
              # below score is DeepLearning score
              avg_anchor_score
              std_anchor_score
              anchor_score1
              anchor_score2
              avg_ocr_score
              std_ocr_score
              ocr_score1
              ocr_score2
@author: sjlgg
'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from parameters import POS_LOOP_PATH, NEG_LOOP_PATH, CELL_LINE_LIST, LOOP_MODEL_PATH, FINE_TUNED_OCR_MODEL_PATH, FINE_TUNED_ANCHOR_MODEL_PATH, PRETRAINED_MODEL_PATH
from pretrained_model import anchor_predictor
from fine_tuned_ocr_model import ocr_predictor, ocr_models
from pretrained_model.pytorch_glove.tools import KMerDictionary

def preprocess_loop(cell_line_list, anchor_model, ocr_model, kmd, device):
    '''
    生成环的训练数据
    '''
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        out_file = open(LOOP_MODEL_PATH+"/ml_training_data/{}.loop".format(cell_line), 'w')
        out_file.write("chrom\tstart1\tstart2\tresponse\tlength\tmotif_pattern\tavg_motif_strength\tstd_motif_strength\tavg_conservation\tstd_conservation\tavg_HS\tstd_HS\tHS_in-between\tHS_left\tHS_right\tavg_anchor_score\tstd_anchor_score\tanchor_score1\tanchor_score2\tavg_ocr_score\tstd_ocr_score\tocr_score1\tocr_score2\n")
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            f = open(pos_loop_path, 'r')
            lines = f.readlines()
            f.close()
            pos_total_sample_num += len(lines)
            for line in lines:
                fields = line.strip().split()
                chrom = fields[0]
                start1 = int(float(fields[4]))
                start2 = int(float(fields[10]))
                response = 1
                length = start2 - start1
                motif_pattern = 5 - i
                avg_motif_strength = fields[12]
                std_motif_strength = fields[13]
                avg_conservation = fields[14]
                std_conservation = fields[15]
                avg_HS = fields[16]
                std_HS = fields[17]
                HS_in_between = fields[18]
                HS_left = fields[19]
                HS_right = fields[20]
                # DL score
                seq1 = fields[5]
                seq2 = fields[11]
                if "N" in seq1 or "N" in seq2: continue
                kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                std_anchor_score = np.std([anchor_score1, anchor_score2])
                ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0][0].item()
                ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0][0].item()
                avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                std_ocr_score = np.std([ocr_score1, ocr_score2])
                out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
        neg_sample_num = pos_total_sample_num // 12
        for typee in range(0, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                f = open(neg_loop_path, 'r')
                lines = f.readlines()
                f.close()
                lines = lines[:neg_sample_num]
                for line in lines:
                    fields = line.strip().split()
                    chrom = fields[0]
                    start1 = int(float(fields[4]))
                    start2 = int(float(fields[10]))
                    response = 0
                    length = start2 - start1
                    motif_pattern = judge_loop_type([fields[3], fields[9]])
                    avg_motif_strength = fields[12]
                    std_motif_strength = fields[13]
                    avg_conservation = fields[14]
                    std_conservation = fields[15]
                    avg_HS = fields[16]
                    std_HS = fields[17]
                    HS_in_between = fields[18]
                    HS_left = fields[19]
                    HS_right = fields[20]
                    # DL score
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2: continue
                    kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                    kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                    anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                    anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                    avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                    std_anchor_score = np.std([anchor_score1, anchor_score2])
                    ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0][0].item()
                    ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0][0].item()
                    avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                    std_ocr_score = np.std([ocr_score1, ocr_score2])
                    out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
        out_file.close()

def preprocess_loop_v2(cell_line_list, anchor_model, ocr_model, kmd, device):
    '''
    生成环的训练数据，
    与v1的区别是v2的OCR数据是基于DNase/{cell_line}/{cell_line}_OCR_ext2.bed中的
    label为1000个碱基signalValue的数据制作的
    '''
    # ! 注意由于产生neg loop过程中设计乱序等操作，所以某一些neg loop的位置是错误的。这些loop的anchor不能拿来当样本。
    # ! 下面列举每个类型的数据情况:
    '''
    pos loop的五类都是可以的，因为都是从正常正例里抽取的。
    neg loop共12类。
    type_0: # ! 有问题，anchor1的序列是打乱的，所以位置对应不上，也就没有signalValue的数据
    type_1: 没问题，这个环的两个锚点是pos motif anchor & neg motif anchor。
    type_2: 没问题，后续各种类型见negative_loop_generator.py。总之就是1-5都没问题，都是序列和位置是对应的。
    type_3: type_4: type_5: 
    所以下面neg loop的循环直接从1开始。不要0了。
    '''
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        out_file = open(LOOP_MODEL_PATH+"/ml_training_data/{}.loop".format(cell_line), 'w')
        out_file.write("chrom\tstart1\tstart2\tresponse\tlength\tmotif_pattern\tavg_motif_strength\tstd_motif_strength\tavg_conservation\tstd_conservation\tavg_HS\tstd_HS\tHS_in-between\tHS_left\tHS_right\tavg_anchor_score\tstd_anchor_score\tanchor_score1\tanchor_score2\tavg_ocr_score\tstd_ocr_score\tocr_score1\tocr_score2\n")
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            f = open(pos_loop_path, 'r')
            lines = f.readlines()
            f.close()
            pos_total_sample_num += len(lines)
            for line in lines:
                fields = line.strip().split()
                chrom = fields[0]
                start1 = int(float(fields[4]))
                start2 = int(float(fields[10]))
                response = 1
                length = start2 - start1
                motif_pattern = 5 - i
                avg_motif_strength = fields[12]
                std_motif_strength = fields[13]
                avg_conservation = fields[14]
                std_conservation = fields[15]
                avg_HS = fields[16]
                std_HS = fields[17]
                HS_in_between = fields[18]
                HS_left = fields[19]
                HS_right = fields[20]
                # DL score
                seq1 = fields[5]
                seq2 = fields[11]
                if "N" in seq1 or "N" in seq2: continue
                kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                std_anchor_score = np.std([anchor_score1, anchor_score2])
                ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0].mean().item()
                ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0].mean().item()
                avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                std_ocr_score = np.std([ocr_score1, ocr_score2])
                out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
        neg_sample_num = pos_total_sample_num // 10
        for typee in range(1, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                f = open(neg_loop_path, 'r')
                lines = f.readlines()
                f.close()
                lines = lines[:neg_sample_num]
                for line in lines:
                    fields = line.strip().split()
                    chrom = fields[0]
                    start1 = int(float(fields[4]))
                    start2 = int(float(fields[10]))
                    response = 0
                    length = start2 - start1
                    motif_pattern = judge_loop_type([fields[3], fields[9]])
                    avg_motif_strength = fields[12]
                    std_motif_strength = fields[13]
                    avg_conservation = fields[14]
                    std_conservation = fields[15]
                    avg_HS = fields[16]
                    std_HS = fields[17]
                    HS_in_between = fields[18]
                    HS_left = fields[19]
                    HS_right = fields[20]
                    # DL score
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2: continue
                    kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                    kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                    anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                    anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                    avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                    std_anchor_score = np.std([anchor_score1, anchor_score2])
                    ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0].mean().item()
                    ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0].mean().item()
                    avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                    std_ocr_score = np.std([ocr_score1, ocr_score2])
                    out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
        out_file.close()

def preprocess_loop_dl(cell_line_list, kmd):
    '''
    生成环的训练数据
    '''
    # ! 注意由于产生neg loop过程中设计乱序等操作，所以某一些neg loop的位置是错误的。这些loop的anchor不能拿来当样本。
    # ! 下面列举每个类型的数据情况:
    '''
    pos loop的五类都是可以的，因为都是从正常正例里抽取的。
    neg loop共12类。
    type_0: # ! 有问题，anchor1的序列是打乱的，所以位置对应不上，也就没有signalValue的数据
    type_1: 没问题，这个环的两个锚点是pos motif anchor & neg motif anchor。
    type_2: 没问题，后续各种类型见negative_loop_generator.py。总之就是1-5都没问题，都是序列和位置是对应的。
    type_3: type_4: type_5: 
    所以下面neg loop的循环直接从1开始。不要0了。
    '''
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        out_file = open(LOOP_MODEL_PATH+"/dl_training_data/{}.loop".format(cell_line), 'w')
        out_file.write("chrom\tstart1\tstart2\tresponse\tlength\tmotif_pattern\tavg_motif_strength\tstd_motif_strength\tavg_conservation\tstd_conservation\n")
        kmers_list1, kmers_list2 = list(), list()
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            f = open(pos_loop_path, 'r')
            lines = f.readlines()
            f.close()
            pos_total_sample_num += len(lines)
            for line in lines:
                fields = line.strip().split()
                chrom = fields[0]
                start1 = int(float(fields[4]))
                start2 = int(float(fields[10]))
                response = 1
                length = start2 - start1
                motif_pattern = 5 - i
                avg_motif_strength = fields[12]
                std_motif_strength = fields[13]
                avg_conservation = fields[14]
                std_conservation = fields[15]
                out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\n")
                # DL score
                seq1 = fields[5]
                seq2 = fields[11]
                kmers1 = kmd.gen_kmers(seq1)
                kmers2 = kmd.gen_kmers(seq2)
                kmers_list1.append(kmers1)
                kmers_list2.append(kmers2)
        neg_sample_num = pos_total_sample_num // 10
        for typee in range(1, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                f = open(neg_loop_path, 'r')
                lines = f.readlines()
                f.close()
                lines = lines[:neg_sample_num]
                for line in lines:
                    fields = line.strip().split()
                    chrom = fields[0]
                    start1 = int(float(fields[4]))
                    start2 = int(float(fields[10]))
                    response = 0
                    length = start2 - start1
                    motif_pattern = judge_loop_type([fields[3], fields[9]])
                    avg_motif_strength = fields[12]
                    std_motif_strength = fields[13]
                    avg_conservation = fields[14]
                    std_conservation = fields[15]
                    out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\n")
                    # DL score
                    seq1 = fields[5]
                    seq2 = fields[11]
                    kmers1 = kmd.gen_kmers(seq1)
                    kmers2 = kmd.gen_kmers(seq2)
                    kmers_list1.append(kmers1)
                    kmers_list2.append(kmers2)
        out_file.close()
        kmers_list1 = np.array(kmers_list1)
        kmers_list2 = np.array(kmers_list2)
        torch.save([kmers_list1, kmers_list2], LOOP_MODEL_PATH+"/dl_training_data/{}.kmers".format(cell_line))

def preprocess_loop_dl_v2(cell_line_list, anchor_model, ocr_model, kmd, device):
    '''
    生成环的训练数据
    '''
    # ! 注意由于产生neg loop过程中设计乱序等操作，所以某一些neg loop的位置是错误的。这些loop的anchor不能拿来当样本。
    # ! 下面列举每个类型的数据情况:
    '''
    pos loop的五类都是可以的，因为都是从正常正例里抽取的。
    neg loop共12类。
    type_0: # ! 有问题，anchor1的序列是打乱的，所以位置对应不上，也就没有signalValue的数据
    type_1: 没问题，这个环的两个锚点是pos motif anchor & neg motif anchor。
    type_2: 没问题，后续各种类型见negative_loop_generator.py。总之就是1-5都没问题，都是序列和位置是对应的。
    type_3: type_4: type_5: 
    所以下面neg loop的循环直接从1开始。不要0了。
    '''
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        out_file = open(LOOP_MODEL_PATH+"/dl_training_data/{}.loop_v2".format(cell_line), 'w')
        out_file.write("chrom\tstart1\tstart2\tresponse\tlength\tmotif_pattern\tavg_motif_strength\tstd_motif_strength\tavg_conservation\tstd_conservation\tavg_anchor_score\tstd_anchor_score\tanchor_score1\tanchor_score2\tavg_ocr_score\tstd_ocr_score\tocr_score1\tocr_score2\n")
        kmers_list1, kmers_list2 = list(), list()
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            print("=。=", pos_loop_path)
            f = open(pos_loop_path, 'r')
            lines = f.readlines()
            f.close()
            pos_total_sample_num += len(lines)
            for line in lines:
                fields = line.strip().split()
                chrom = fields[0]
                start1 = int(float(fields[4]))
                start2 = int(float(fields[10]))
                response = 1
                length = start2 - start1
                motif_pattern = 5 - i
                avg_motif_strength = fields[12]
                std_motif_strength = fields[13]
                avg_conservation = fields[14]
                std_conservation = fields[15]
                # DL score
                seq1 = fields[5]
                seq2 = fields[11]
                if "N" in seq1 or "N" in seq2: continue
                kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                std_anchor_score = np.std([anchor_score1, anchor_score2])
                ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0].mean().item()
                ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0].mean().item()
                avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                std_ocr_score = np.std([ocr_score1, ocr_score2])
                out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
                # seq feature
                kmers1 = kmd.gen_kmers(seq1)
                kmers2 = kmd.gen_kmers(seq2)
                kmers_list1.append(kmers1)
                kmers_list2.append(kmers2)
        neg_sample_num = pos_total_sample_num // 10
        for typee in range(1, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                print("=。=", neg_loop_path)
                f = open(neg_loop_path, 'r')
                lines = f.readlines()
                f.close()
                lines = lines[:neg_sample_num]
                for line in lines:
                    fields = line.strip().split()
                    chrom = fields[0]
                    start1 = int(float(fields[4]))
                    start2 = int(float(fields[10]))
                    response = 0
                    length = start2 - start1
                    motif_pattern = judge_loop_type([fields[3], fields[9]])
                    avg_motif_strength = fields[12]
                    std_motif_strength = fields[13]
                    avg_conservation = fields[14]
                    std_conservation = fields[15]
                    # DL score
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2: continue
                    kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
                    kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
                    anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
                    anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
                    avg_anchor_score = (anchor_score1 + anchor_score2) / 2
                    std_anchor_score = np.std([anchor_score1, anchor_score2])
                    ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0].mean().item()
                    ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0].mean().item()
                    avg_ocr_score = (ocr_score1 + ocr_score2) / 2
                    std_ocr_score = np.std([ocr_score1, ocr_score2])
                    out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_anchor_score}\t{std_anchor_score}\t{anchor_score1}\t{anchor_score2}\t{avg_ocr_score}\t{std_ocr_score}\t{ocr_score1}\t{ocr_score2}\n")
                    # seq feature
                    kmers1 = kmd.gen_kmers(seq1)
                    kmers2 = kmd.gen_kmers(seq2)
                    kmers_list1.append(kmers1)
                    kmers_list2.append(kmers2)
        out_file.close()
        kmers_list1 = np.array(kmers_list1)
        kmers_list2 = np.array(kmers_list2)
        torch.save([kmers_list1, kmers_list2], LOOP_MODEL_PATH+"/dl_training_data/{}.kmers_v2".format(cell_line))

def load_loop_dl_data(cell_line_list, val_chr_list=['chr22', 'chrX']):
    train_features = np.array([])
    train_kmers1 = np.array([])
    train_kmers2 = np.array([])
    train_labels = np.array([])
    val_features = np.array([])
    val_kmers1 = np.array([])
    val_kmers2 = np.array([])
    val_labels = np.array([])
    for cell_line in cell_line_list:
        # chrom|start1|start2|response|length|motif_pattern|avg_motif_strength|std_motif_strength|avg_conservation|std_conservation
        ml_data = pd.read_table(LOOP_MODEL_PATH+"/dl_training_data/{}.loop_v2".format(cell_line), sep='\t')
        # every element is a kmer list for a 1000bp sequence
        kmers_list1, kmers_list2 = torch.load(LOOP_MODEL_PATH+"/dl_training_data/{}.kmers_v2".format(cell_line))
        assert len(ml_data) == len(kmers_list1) == len(kmers_list2), "[ERROR]! Different number of samples in ml_data and kmers_list"

        train_indices = ml_data['chrom'].isin(val_chr_list) == False
        val_indices = ml_data['chrom'].isin(val_chr_list) == True
        train_data = ml_data[train_indices]
        val_data = ml_data[val_indices]
        temp_train_features = train_data.iloc[:,4:].to_numpy() # total 6 features in v1, 14 features in v2
        temp_val_features = val_data.iloc[:,4:].to_numpy()
        temp_train_kmers1 = kmers_list1[train_indices]
        temp_val_kmers1 = kmers_list1[val_indices]
        temp_train_kmers2 = kmers_list2[train_indices]
        temp_val_kmers2 = kmers_list2[val_indices]
        temp_train_labels = train_data['response'].to_numpy()
        temp_val_labels = val_data['response'].to_numpy()
        if len(train_features) == 0:
            train_features = temp_train_features
            train_kmers1 = temp_train_kmers1
            train_kmers2 = temp_train_kmers2
            train_labels = temp_train_labels
            val_features = temp_val_features
            val_kmers1 = temp_val_kmers1
            val_kmers2 = temp_val_kmers2
            val_labels = temp_val_labels
        else:
            train_features = np.concatenate((train_features, temp_train_features), axis=0)
            train_kmers1 = np.concatenate((train_kmers1, temp_train_kmers1), axis=0)
            train_kmers2 = np.concatenate((train_kmers2, temp_train_kmers2), axis=0)
            train_labels = np.concatenate((train_labels, temp_train_labels), axis=0)
            val_features = np.concatenate((val_features, temp_val_features), axis=0)
            val_kmers1 = np.concatenate((val_kmers1, temp_val_kmers1), axis=0)
            val_kmers2 = np.concatenate((val_kmers2, temp_val_kmers2), axis=0)
            val_labels = np.concatenate((val_labels, temp_val_labels), axis=0)
    print("[INFO] train_features shape: {}".format(train_features.shape))
    print("[INFO] train_kmers1 shape: {}".format(train_kmers1.shape))
    print("[INFO] train_kmers2 shape: {}".format(train_kmers2.shape))
    print("[INFO] train_labels shape: {}".format(train_labels.shape))
    print("[INFO] val_features shape: {}".format(val_features.shape))
    print("[INFO] val_kmers1 shape: {}".format(val_kmers1.shape))
    print("[INFO] val_kmers2 shape: {}".format(val_kmers2.shape))
    print("[INFO] val_labels shape: {}".format(val_labels.shape))
    return train_features, train_kmers1, train_kmers2, train_labels, val_features, val_kmers1, val_kmers2, val_labels

def load_loop_dl_data_by_cell_line(cell_line_list, val_cell_line_list):
    train_features = np.array([])
    train_kmers1 = np.array([])
    train_kmers2 = np.array([])
    train_labels = np.array([])
    val_features = np.array([])
    val_kmers1 = np.array([])
    val_kmers2 = np.array([])
    val_labels = np.array([])
    for cell_line in cell_line_list:
        # chrom|start1|start2|response|length|motif_pattern|avg_motif_strength|std_motif_strength|avg_conservation|std_conservation
        ml_data = pd.read_table(LOOP_MODEL_PATH+"/dl_training_data/{}.loop_v2".format(cell_line), sep='\t')
        # every element is a kmer list for a 1000bp sequence
        kmers_list1, kmers_list2 = torch.load(LOOP_MODEL_PATH+"/dl_training_data/{}.kmers_v2".format(cell_line))
        assert len(ml_data) == len(kmers_list1) == len(kmers_list2), "[ERROR]! Different number of samples in ml_data and kmers_list"
        
        temp_features = ml_data.iloc[:,4:].to_numpy() # total 6 features in v1, 14 features in v2=
        temp_labels = ml_data['response'].to_numpy()
        if cell_line in val_cell_line_list:
            if len(val_features) == 0:
                val_features = temp_features
                val_kmers1 = kmers_list1
                val_kmers2 = kmers_list2
                val_labels = temp_labels
            else:
                val_features = np.concatenate((val_features, temp_features), axis=0)
                val_kmers1 = np.concatenate((val_kmers1, kmers_list1), axis=0)
                val_kmers2 = np.concatenate((val_kmers2, kmers_list2), axis=0)
                val_labels = np.concatenate((val_labels, temp_labels), axis=0)
        else:
            if len(train_features) == 0:
                train_features = temp_features
                train_kmers1 = kmers_list1
                train_kmers2 = kmers_list2
                train_labels = temp_labels
            else:
                train_features = np.concatenate((train_features, temp_features), axis=0)
                train_kmers1 = np.concatenate((train_kmers1, kmers_list1), axis=0)
                train_kmers2 = np.concatenate((train_kmers2, kmers_list2), axis=0)
                train_labels = np.concatenate((train_labels, temp_labels), axis=0)
    print("[INFO] train_features shape: {}".format(train_features.shape))
    print("[INFO] train_kmers1 shape: {}".format(train_kmers1.shape))
    print("[INFO] train_kmers2 shape: {}".format(train_kmers2.shape))
    print("[INFO] train_labels shape: {}".format(train_labels.shape))
    print("[INFO] val_features shape: {}".format(val_features.shape))
    print("[INFO] val_kmers1 shape: {}".format(val_kmers1.shape))
    print("[INFO] val_kmers2 shape: {}".format(val_kmers2.shape))
    print("[INFO] val_labels shape: {}".format(val_labels.shape))
    return train_features, train_kmers1, train_kmers2, train_labels, val_features, val_kmers1, val_kmers2, val_labels



class LoopDataset(Dataset):
    def __init__(self, features, kmer_data_1, kmer_data_2, labels):
        self.features = torch.tensor(features, dtype=torch.float)
        self.kmer_data_1 = kmer_data_1
        self.kmer_data_2 = kmer_data_2
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.kmer_data_1[index], self.kmer_data_2[index], self.labels[index]

def judge_loop_type(anchor_pair_orientation_strand):
    convergent_anchor_pair = [['f,+','r,+'], ['f,+','f,-'], ['r,-','r,+'], ['r,-','f,-']]
    tandem_anchor_pair = [['f,+','f,+'], ['f,+','r,-'], ['r,+','r,+'], ['r,+','f,-'], ['f,-','r,+'], ['f,-','f,-'], ['r,-','f,+'], ['r,-','r,-']]
    divergent_anchor_pair = [['r,+','f,+'], ['r,+','r,-'], ['f,-','f,+'], ['f,-','r,-']]
    if anchor_pair_orientation_strand == ['.,.', '.,.']:
        return 0
    elif ".,." in anchor_pair_orientation_strand:
        return 1
    elif anchor_pair_orientation_strand in divergent_anchor_pair:
        return 2
    elif anchor_pair_orientation_strand in tandem_anchor_pair:
        return 3
    elif anchor_pair_orientation_strand in convergent_anchor_pair:
        return 4
    else:
        raise ValueError("anchor_pair_orientation_strand is not valid!")

def data_statistics(cell_line):
    '''
    看看正负样本的数据分布
    '''
    loop_data = pd.read_csv(LOOP_MODEL_PATH+"/ml_training_data/{}.loop".format(cell_line), sep='\t')
    pos_loop_data = loop_data[loop_data['response'] == 1]
    neg_loop_data = loop_data[loop_data['response'] == 0]
    # print(loop_data.head())
    print("\tpos\tneg")
    print(f"length:\t{pos_loop_data['length'].mean()}\t{neg_loop_data['length'].mean()}")
    print(f"avg_motif_strength:\t{pos_loop_data['avg_motif_strength'].mean()}\t{neg_loop_data['avg_motif_strength'].mean()}")
    print(f"std_motif_strength:\t{pos_loop_data['std_motif_strength'].mean()}\t{neg_loop_data['std_motif_strength'].mean()}")
    print(f"avg_conservation:\t{pos_loop_data['avg_conservation'].mean()}\t{neg_loop_data['avg_conservation'].mean()}")
    print(f"std_conservation:\t{pos_loop_data['std_conservation'].mean()}\t{neg_loop_data['std_conservation'].mean()}")
    print(f"avg_HS:\t{pos_loop_data['avg_HS'].mean()}\t{neg_loop_data['avg_HS'].mean()}")
    print(f"std_HS:\t{pos_loop_data['std_HS'].mean()}\t{neg_loop_data['std_HS'].mean()}")
    print(f"HS_in_between:\t{pos_loop_data['HS_in-between'].mean()}\t{neg_loop_data['HS_in-between'].mean()}")
    print(f"HS_left:\t{pos_loop_data['HS_left'].mean()}\t{neg_loop_data['HS_left'].mean()}")
    print(f"HS_right:\t{pos_loop_data['HS_right'].mean()}\t{neg_loop_data['HS_right'].mean()}")
    print(f"avg_anchor_score:\t{pos_loop_data['avg_anchor_score'].mean()}\t{neg_loop_data['avg_anchor_score'].mean()}")
    print(f"std_anchor_score:\t{pos_loop_data['std_anchor_score'].mean()}\t{neg_loop_data['std_anchor_score'].mean()}")
    print(f"anchor_score1:\t{pos_loop_data['anchor_score1'].mean()}\t{neg_loop_data['anchor_score1'].mean()}")
    print(f"anchor_score2:\t{pos_loop_data['anchor_score2'].mean()}\t{neg_loop_data['anchor_score2'].mean()}")
    print(f"avg_ocr_score:\t{pos_loop_data['avg_ocr_score'].mean()}\t{neg_loop_data['avg_ocr_score'].mean()}")
    print(f"std_ocr_score:\t{pos_loop_data['std_ocr_score'].mean()}\t{neg_loop_data['std_ocr_score'].mean()}")
    print(f"ocr_score1:\t{pos_loop_data['ocr_score1'].mean()}\t{neg_loop_data['ocr_score1'].mean()}")
    print(f"ocr_score2:\t{pos_loop_data['ocr_score2'].mean()}\t{neg_loop_data['ocr_score2'].mean()}")
    # distribution of HS and ocr score
    for i in range(20):
        ind = i/10
        print(f"{ind:.1f}-{ind+0.1:.1f}",
              len(pos_loop_data[(pos_loop_data['HS_left']<ind+0.1) & (pos_loop_data['HS_left']>=ind)]),
              len(pos_loop_data[(pos_loop_data['HS_right']<ind+0.1) & (pos_loop_data['HS_right']>=ind)]),
              len(neg_loop_data[(neg_loop_data['HS_left']<ind+0.1) & (neg_loop_data['HS_left']>=ind)]),
              len(neg_loop_data[(neg_loop_data['HS_right']<ind+0.1) & (neg_loop_data['HS_right']>=ind)]),
              len(pos_loop_data[(pos_loop_data['ocr_score1']<ind+0.1) & (pos_loop_data['ocr_score1']>=ind)]),
              len(pos_loop_data[(pos_loop_data['ocr_score2']<ind+0.1) & (pos_loop_data['ocr_score2']>=ind)]),
              len(neg_loop_data[(neg_loop_data['ocr_score1']<ind+0.1) & (neg_loop_data['ocr_score1']>=ind)]),
              len(neg_loop_data[(neg_loop_data['ocr_score2']<ind+0.1) & (neg_loop_data['ocr_score2']>=ind)]))


if __name__ == "__main__":

    #### generate ml data
    # 4cell line : PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_06_24_15_23_42_GM12878_AnchorModel_v20230516_v1.pt'
    #              FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_06_24_21_51_34_GM12878_OCR_ext2_OCRModel_v20230524_v1.pt'
    # 8 cell line by chr: PretrainedModel_2023_06_29_20_59_01.pt
    #                     OCRModel_val_loss_2023_06_30_23_34_50.pt
    # 8 cell line health/cancer : PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt
    #                             OCRModel_val_loss_2023_07_07_19_04_18.pt
    anchor_model, device, kmd = anchor_predictor.init_predictor(model_path=PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt')
    ocr_model, _, _ = ocr_predictor.init_predictor(
                          model_structure=ocr_models.OCRModel_v20230524_v1(),
                          model_path=FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_07_07_19_04_18.pt'
                      )
    # preprocess_loop_v2(["GM12878", "HCT116", "IMR-90", "K562"], anchor_model, ocr_model, kmd, device)
    # preprocess_loop_v2(["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"], anchor_model, ocr_model, kmd, device)

    # data_statistics("OCRModel_val_loss_2023_05_29_15_43_07_GM12878_OCR_ext2_OCRModel_v20230524_v1")

    #### generate dl data
    kmd_5 = KMerDictionary(kmer_len=5)
    # preprocess_loop_dl(["GM12878"], kmd_5)
    # preprocess_loop_dl_v2(["GM12878", "HCT116", "IMR-90", "K562"], anchor_model, ocr_model, kmd_5, device)
    preprocess_loop_dl_v2(["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"], anchor_model, ocr_model, kmd_5, device)
    # load_loop_dl_data(cell_line_list=["GM12878"], val_chr_list=['chr22', 'chrX'])

