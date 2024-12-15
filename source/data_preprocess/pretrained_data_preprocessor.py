"""
@author: sjl
@purpose: prepare pretrained data
"""

import sys, os, glob, copy, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import PRETRAINED_DATA_PATH, CHIA_PET2_RESULT_PATH, CHIP_SEQ_PATH, CHROM_SIZES, DNASE_DIR, CELL_LINE_LIST
from loop_preprocessor import load_UCSC_hg38
from collections import defaultdict
from utils import TimeStatistic

@TimeStatistic
def gen_pretrained_data(cell_line_list):
    """
    generate pretrained data
    """
    print("generating pretrained data...")
    chrom_sizes = dict()
    with open(CHROM_SIZES, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields[0]) < 6:
                chrom_sizes[fields[0]] = int(fields[1])
    CHROMOSOME_DICT = load_UCSC_hg38()
    for cell_line in cell_line_list:
        print(f"generating {cell_line} pretrained data...")
        # final output data
        pos_anchor_ocr_list = list()
        pos_anchor_ccr_list = list()
        neg_anchor_ocr_list = list()
        neg_anchor_ccr_list = list()
        # data from chia-pet and chip-seq
        pos_anchor_dict = defaultdict(list) # must be pos
        candidate_anchor_dict = defaultdict(list) # for get non_anchor_list, this list of anchor may be pos but dont enrich in both chia-pet & chip-seq
        non_anchor_region_dict = defaultdict(list) # for get non_anchor_list

        # get pos_anchor_dict and candidate_anchor_dict
        # read chia-pet anchors
        chia_pet_path = CHIA_PET2_RESULT_PATH.format(cell_line)+\
            '{}.significant.interactions.intra.filtered.MICC'.format(cell_line)
        chia_pet_anchor_dict = defaultdict(list)
        with open(chia_pet_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields[0]) > 5: continue
                chia_pet_anchor_dict[fields[0]].append((int(fields[1]), int(fields[2])))
                chia_pet_anchor_dict[fields[3]].append((int(fields[4]), int(fields[5])))
        for key in chia_pet_anchor_dict: # 按chr分组
            chia_pet_anchor_dict[key] = list([anchor[0], anchor[1]] for anchor in set(chia_pet_anchor_dict[key]))
            chia_pet_anchor_dict[key].sort(key=lambda x: (int(x[0]), int(x[1])))

        # read chip-seq anchors
        chip_seq_path = CHIP_SEQ_PATH+f"/{cell_line}/"
        chip_seq_path = glob.glob(chip_seq_path+r'*.bed')[0]
        chip_seq_anchor_dict = defaultdict(list)
        with open(chip_seq_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields[0]) > 5: continue
                chip_seq_anchor_dict[fields[0]].append((int(fields[1]), int(fields[2])))
        for key in chip_seq_anchor_dict:
            chip_seq_anchor_dict[key] = list([anchor[0], anchor[1]] for anchor in set(chip_seq_anchor_dict[key]))
            chip_seq_anchor_dict[key].sort(key=lambda x: (int(x[0]), int(x[1])))

        # get chromatin accessible region
        dnase_path = DNASE_DIR+f"/{cell_line}/"
        dnase_path = glob.glob(dnase_path+r'*.filtered.bed')[0]
        dnase_ocr_dict = defaultdict(list)
        with open(dnase_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields[0]) > 5: continue
                dnase_ocr_dict[fields[0]].append((int(fields[1]), int(fields[2])))
        # closed chromatin region
        dnase_ccr_dict = defaultdict(list)

        # get overlapped anchors by chr
        chrs = list()
        chrs.extend(chia_pet_anchor_dict.keys())
        chrs.extend(chip_seq_anchor_dict.keys())
        chrs.extend(dnase_ocr_dict.keys())
        chrs = list(set(chrs))
        chrs.sort(key=lambda x: int(x[3:]) if x[3:].isdigit() else ord(x[3:]))
        for chr in chrs:
            overlapped_anchor_list = get_overlapped_anchor_list(connect_connected_region(chia_pet_anchor_dict[chr]), connect_connected_region(chip_seq_anchor_dict[chr]))
            pos_anchor_dict[chr] = [(anchor[0], anchor[1]) for anchor in overlapped_anchor_list if anchor[1]-anchor[0]>=50]
            # for non anchor region, get non_anchor_region_dict
            temp_anchor_list = chia_pet_anchor_dict[chr]+chip_seq_anchor_dict[chr]
            temp_anchor_list.sort(key=lambda x: (int(x[0]), int(x[1])))
            candidate_anchor_dict[chr] = connect_connected_region(temp_anchor_list)
            non_anchor_region_dict[chr] = [(region[0], region[1]) for region in get_complement(chrom_sizes[chr], candidate_anchor_dict[chr]) if region[1]-region[0]>=500]
            dnase_ccr_dict[chr] = [(region[0], region[1]) for region in get_complement(chrom_sizes[chr], dnase_ocr_dict[chr]) if region[1]-region[0]>=500]

        # // statistics dnase length
        max_length = 0
        min_length = 99999999999999
        region_number = 0
        for chr in chrs:
            region_number += len(dnase_ocr_dict[chr])
            for anchor in dnase_ocr_dict[chr]:
                t = anchor[1]-anchor[0]
                if t > max_length:
                    max_length = t
                if t < min_length:
                    min_length = t
        print(f"dnase ocr num: {region_number}, [{min_length}, {max_length}]")
        # // statistics dnase closed chromatin region length
        max_length = 0
        min_length = 99999999999999
        region_number = 0
        for chr in chrs:
            region_number += len(dnase_ccr_dict[chr])
            for anchor in dnase_ccr_dict[chr]:
                t = anchor[1]-anchor[0]
                if t > max_length:
                    max_length = t
                if t < min_length:
                    min_length = t
        print(f"dnase ccr num: {region_number}, [{min_length}, {max_length}]")
        # // statistics 临时看看长度
        max_length = 0
        min_length = 99999999999999
        region_number = 0
        for chr in chrs:
            region_number += len(pos_anchor_dict[chr])
            for anchor in pos_anchor_dict[chr]:
                t = anchor[1]-anchor[0]
                if t > max_length:
                    max_length = t
                if t < min_length:
                    min_length = t
        print(f"pos_anchor_region num: {region_number}, [{min_length}, {max_length}]")
        max_length = 0
        min_length = 99999999999999
        region_number = 0
        for chr in chrs:
            region_number += len(non_anchor_region_dict[chr])
            for anchor in non_anchor_region_dict[chr]:
                t = anchor[1]-anchor[0]
                if t > max_length:
                    max_length = t
                if t < min_length:
                    min_length = t
        print(f"non_anchor_region num: {region_number}, [{min_length}, {max_length}]")

        # fill 4 list of final output data
        OVERLAP_RATIO_THRESHOLD = 0.8
        # fill pos_anchor_ocr_list
        for chr in chrs:
            for pos_anchor in pos_anchor_dict[chr]:
                for anchor_ocr in dnase_ocr_dict[chr]:
                    region = compute_overlap_ratio(region_1=(pos_anchor[0], pos_anchor[1]), region_2=(anchor_ocr[0], anchor_ocr[1]), overlap_ratio_threshold=OVERLAP_RATIO_THRESHOLD)
                    if region and region[1]-region[0]>=50:
                        pos_anchor_ocr_list.append([chr, region[0], region[1]])
        # // statistics
        pos_anchor_ocr_list_len = len(pos_anchor_ocr_list)
        max_length = 0
        min_length = 99999999999999
        for anchor in pos_anchor_ocr_list:
            t = anchor[2]-anchor[1]
            if t > max_length:
                max_length = t
            if t < min_length:
                min_length = t
        print(f"pos_anchor_ocr_list len: {pos_anchor_ocr_list_len}, [{min_length}, {max_length}]")
        # fill pos_anchor_ccr_list
        for chr in chrs:
            for pos_anchor in pos_anchor_dict[chr]:
                for anchor_ocr in dnase_ccr_dict[chr]:
                    region = compute_overlap_ratio(region_1=(pos_anchor[0], pos_anchor[1]), region_2=(anchor_ocr[0], anchor_ocr[1]), overlap_ratio_threshold=OVERLAP_RATIO_THRESHOLD)
                    if region and region[1]-region[0]>=50:
                        pos_anchor_ccr_list.append([chr, region[0], region[1]])
        # // statistics
        pos_anchor_ccr_list_len = len(pos_anchor_ccr_list)
        max_length = 0
        min_length = 99999999999999
        for anchor in pos_anchor_ccr_list:
            t = anchor[2]-anchor[1]
            if t > max_length:
                max_length = t
            if t < min_length:
                min_length = t
        print(f"pos_anchor_ccr_list len: {pos_anchor_ccr_list_len}, [{min_length}, {max_length}]")
        # fill neg_anchor_ocr_list
        for chr in chrs:
            for pos_anchor in non_anchor_region_dict[chr]:
                for anchor_ocr in dnase_ocr_dict[chr]:
                    region = compute_overlap_ratio(region_1=(pos_anchor[0], pos_anchor[1]), region_2=(anchor_ocr[0], anchor_ocr[1]), overlap_ratio_threshold=OVERLAP_RATIO_THRESHOLD)
                    if region and region[1]-region[0]>=50:
                        neg_anchor_ocr_list.append([chr, region[0], region[1]])
        # // statistics
        neg_anchor_ocr_list_len = len(neg_anchor_ocr_list)
        max_length = 0
        min_length = 99999999999999
        for anchor in neg_anchor_ocr_list:
            t = anchor[2]-anchor[1]
            if t > max_length:
                max_length = t
            if t < min_length:
                min_length = t
        print(f"neg_anchor_ocr_list len: {neg_anchor_ocr_list_len}, [{min_length}, {max_length}]")
        # fill neg_anchor_ccr_list
        for chr in chrs:
            for pos_anchor in non_anchor_region_dict[chr]:
                for anchor_ocr in dnase_ccr_dict[chr]:
                    region = compute_overlap_ratio(region_1=(pos_anchor[0], pos_anchor[1]), region_2=(anchor_ocr[0], anchor_ocr[1]), overlap_ratio_threshold=OVERLAP_RATIO_THRESHOLD)
                    if region and region[1]-region[0]>=50:
                        neg_anchor_ccr_list.append([chr, region[0], region[1]])
        # // statistics
        neg_anchor_ccr_list_len = len(neg_anchor_ccr_list)
        max_length = 0
        min_length = 99999999999999
        for anchor in neg_anchor_ccr_list:
            t = anchor[2]-anchor[1]
            if t > max_length:
                max_length = t
            if t < min_length:
                min_length = t
        print(f"neg_anchor_ccr_list len: {neg_anchor_ccr_list_len}, [{min_length}, {max_length}]")

        sample_num = min(pos_anchor_ocr_list_len, pos_anchor_ccr_list_len, neg_anchor_ocr_list_len, neg_anchor_ccr_list_len)

        random.seed(9523)
        pos_anchor_ocr_out_list = list()
        pos_anchor_ccr_out_list = list()
        neg_anchor_ocr_out_list = list()
        neg_anchor_ccr_out_list = list()
        while len(pos_anchor_ocr_out_list) < sample_num and len(pos_anchor_ocr_list) > 0:
            region = pos_anchor_ocr_list.pop(random.randint(0, len(pos_anchor_ocr_list)-1))
            extend_feature(CHROMOSOME_DICT, region)
            if "N" not in region[3] and len(region[3]) == 1000:
                pos_anchor_ocr_out_list.append(region)
        while len(pos_anchor_ccr_out_list) < sample_num and len(pos_anchor_ccr_list) > 0:
            region = pos_anchor_ccr_list.pop(random.randint(0, len(pos_anchor_ccr_list)-1))
            extend_feature(CHROMOSOME_DICT, region)
            if "N" not in region[3] and len(region[3]) == 1000:
                pos_anchor_ccr_out_list.append(region)
        while len(neg_anchor_ocr_out_list) < sample_num and len(neg_anchor_ocr_list) > 0:
            region = neg_anchor_ocr_list.pop(random.randint(0, len(neg_anchor_ocr_list)-1))
            extend_feature(CHROMOSOME_DICT, region)
            if "N" not in region[3] and len(region[3]) == 1000:
                neg_anchor_ocr_out_list.append(region)
        while len(neg_anchor_ccr_out_list) < sample_num and len(neg_anchor_ccr_list) > 0:
            region = neg_anchor_ccr_list.pop(random.randint(0, len(neg_anchor_ccr_list)-1))
            extend_feature(CHROMOSOME_DICT, region)
            if "N" not in region[3] and len(region[3]) == 1000:
                neg_anchor_ccr_out_list.append(region)
        print(len(pos_anchor_ocr_out_list))
        print(len(pos_anchor_ccr_out_list))
        print(len(neg_anchor_ocr_out_list))
        print(len(neg_anchor_ccr_out_list))
        pos_anchor_ocr_out_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
        pos_anchor_ccr_out_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
        neg_anchor_ocr_out_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
        neg_anchor_ccr_out_list.sort(key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
        with open(PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ocr.csv", 'w') as f:
            for region in pos_anchor_ocr_out_list:
                f.write('\t'.join(map(str, region))+'\n')
        with open(PRETRAINED_DATA_PATH+f"/{cell_line}_pos_anchor_ccr.csv", 'w') as f:
            for region in pos_anchor_ccr_out_list:
                f.write('\t'.join(map(str, region))+'\n')
        with open(PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ocr.csv", 'w') as f:
            for region in neg_anchor_ocr_out_list:
                f.write('\t'.join(map(str, region))+'\n')
        with open(PRETRAINED_DATA_PATH+f"/{cell_line}_neg_anchor_ccr.csv", 'w') as f:
            for region in neg_anchor_ccr_out_list:
                f.write('\t'.join(map(str, region))+'\n')

def connect_connected_region(anchor_list):
    anchor_list = copy.deepcopy(anchor_list)
    new_anchor_list = list()
    while len(anchor_list) > 0:
        temp_anchor = anchor_list.pop(0)
        while len(anchor_list) > 0 and temp_anchor[1] >= anchor_list[0][0]:
            # print(temp_anchor, anchor_list[0])
            if temp_anchor[1] <= anchor_list[0][1]:
                temp_anchor[1] = anchor_list[0][1]
            anchor_list.pop(0)
        new_anchor_list.append(temp_anchor)
    return new_anchor_list

def get_overlapped_anchor_list(chia_pet_anchor_list, chip_seq_anchor_list):
    overlapped_anchor_list = list()
    while len(chia_pet_anchor_list) > 0 and len(chip_seq_anchor_list) > 0:
        chia_pet_anchor = chia_pet_anchor_list[0]
        chip_seq_anchor = chip_seq_anchor_list[0]
        if chip_seq_anchor[0] <= chia_pet_anchor[0] <= chip_seq_anchor[1]:
            if chia_pet_anchor[1] <= chip_seq_anchor[1]:
                overlapped_anchor_list.append([chia_pet_anchor[0], chia_pet_anchor[1]])
                chia_pet_anchor_list.pop(0)
            elif chia_pet_anchor[1] > chip_seq_anchor[1]:
                overlapped_anchor_list.append([chia_pet_anchor[0], chip_seq_anchor[1]])
                chip_seq_anchor_list.pop(0)
        elif chia_pet_anchor[0] <= chip_seq_anchor[0] <= chia_pet_anchor[1]:
            if chip_seq_anchor[1] <= chia_pet_anchor[1]:
                overlapped_anchor_list.append([chip_seq_anchor[0], chip_seq_anchor[1]])
                chip_seq_anchor_list.pop(0)
            elif chip_seq_anchor[1] > chia_pet_anchor[1]:
                overlapped_anchor_list.append([chip_seq_anchor[0], chia_pet_anchor[1]])
                chia_pet_anchor_list.pop(0)
        elif chia_pet_anchor[1] < chip_seq_anchor[0]:
            chia_pet_anchor_list.pop(0)
        elif chip_seq_anchor[1] < chia_pet_anchor[0]:
            chip_seq_anchor_list.pop(0)
    return overlapped_anchor_list

# 定义一个函数，输入一个序列的长度和一个区域列表，输出不在区域列表中的区域列表
def get_complement(chrom_size, region_list):
    # 初始化一个空列表
    complement = []
    # 初始化一个起始点
    start = 0
    # 遍历区域列表中的每个区域
    for region in region_list:
        # 如果区域的开始大于起始点，说明有一个不在区域列表中的区域
        if region[0] > start:
            end = region[0] # 结束点设为区域的开始减1
            complement.append([start, end]) # 把区域加入到列表中
        # 更新起始点为区域的结束加1
        start = region[1]
    # 如果起始点小于等于序列的长度，说明还有一个不在区域列表中的区域
    if start <= chrom_size:
        end = chrom_size # 结束点设为序列的长度
        complement.append([start, end]) # 把区域加入到列表中, non-motif的orientation和strand设置为"."
    # 返回区域列表
    return complement

def compute_overlap_ratio(region_1, region_2, overlap_ratio_threshold=0.8):
    """
    计算两个区域的重叠部分占两个区域中更小的区域的大小比例。

    参数：
    region_1: 包含两个整数的列表，表示第一个区域的起始位置和结束位置。
    region_2: 包含两个整数的列表，表示第二个区域的起始位置和结束位置。

    返回值：
    重叠部分占两个区域中更小的区域的大小比例，即交集大小除以更小的区域大小。
    """
    # 计算两个区域的长度和重叠部分的起始位置和结束位置
    length_1 = region_1[1] - region_1[0]
    length_2 = region_2[1] - region_2[0]
    start = max(region_1[0], region_2[0])
    end = min(region_1[1]-1, region_2[1]-1)
    overlap_length = max(0, end - start + 1)

    # 计算重叠部分占两个区域中更小的区域的大小比例
    min_length = min(length_1, length_2)
    overlap_ratio = overlap_length / min_length

    if overlap_ratio >= overlap_ratio_threshold:
        return (start, end+1)
    else:
        return False

def extend_feature(chromosome_dict, region):
    mid_point = round((region[1]+region[2])/2)
    region.append(chromosome_dict[region[0]][mid_point-500: mid_point+500])

if __name__ == '__main__':
    gen_pretrained_data(CELL_LINE_LIST)
    # print(compute_overlap_ratio((1,6), (3,8), overlap_ratio_threshold=0.6))

