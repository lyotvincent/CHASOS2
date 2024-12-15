"""
@author: 孙嘉良
@purpose: generate negative loop samples
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random, copy, time
from parameters import *
from loop_preprocessor import load_CTCF_motifs, load_UCSC_hg38
from utils import TimeStatistic, get_mid_point, shuffle_a_sequence
from collections import defaultdict

@TimeStatistic
def filter_negative_motif(cell_line):
    """
    @purpose: (1) get all CTCF motifs from FIMO result
              (2) get all positive CTCF motifs from *.pos_motif
              (3) all - pos_motif = neg_motif
    @Input: (1)cell line，即要处理哪个cell line的CTCF motif
            (2)FIMO的结果(包含所有CTCF motifs)，FIMO_CUSTOMISED_RESULT_PATH, 注意FIMO结果中的motif是以1为起点的，所以下面将其format即-1
            (3).pos_motif文件(包含已在该cell line的anchors上发现的CTCF motifs)，注意pos_motif已在filter_loop_containing_CTCF函数的生成过程中format(-1)过了
    @Output: (1).neg_motif文件，此cell line全部的negative motif
    """
    # * 由于在此函数前执行了filter_MICC_by_PET_and_FDR，所以此处的pos_motif都是经过这个函数过滤的高质量环上的。
    # * 所以此处生成的neg_motif，既包含未在环上出现的CTCF motif，也包含低质量(PET & FDR in MICC)环上出现的CTCF motif。
    # (1)
    all_CTCF_motifs_dict = load_CTCF_motifs(fimo_file_path=FIMO_CUSTOMISED_RESULT_PATH)
    all_CTCF_motifs_set = set()
    for chr in all_CTCF_motifs_dict.keys():
        for strand in ["+", "-"]:
            for ctcf_start, ctcf_end, orientation in all_CTCF_motifs_dict[chr][strand]:
                formated_ctcf_start, formated_ctcf_end = ctcf_start-1, ctcf_end-1 # 将CTCF motifs的start和end格式化为从0开始
                all_CTCF_motifs_set.add(f"{chr},{formated_ctcf_start},{formated_ctcf_end},{orientation},{strand}")
    # (2)
    positive_CTCF_motifs = set()
    f = open(MOTIF_ON_ANCHOR_PATH.format(cell_line, cell_line), 'r')
    for i in f:
        positive_CTCF_motifs.add(i.strip())
    negative_CTCF_motifs = all_CTCF_motifs_set - positive_CTCF_motifs
    print(f"number: all {len(all_CTCF_motifs_set)}, pos {len(positive_CTCF_motifs)}, neg {len(negative_CTCF_motifs)}")
    # (3)
    negative_CTCF_motifs = sorted(list(negative_CTCF_motifs), key=lambda x: (int(x.split(',')[0][3:]), int(x.split(',')[1])) if x.split(',')[0][3:].isdigit() else (ord(x.split(',')[0][3:]), int(x.split(',')[1])))
    f = open(NEGATIVE_MOTIF_PATH.format(cell_line), 'w')
    for i in negative_CTCF_motifs:
        f.write(i+"\n")
    f.close()

def load_motifs(file_path):
    '''
    @purpose: use for load .pos_motif & .neg_motif
    @return: [[chromosome, start, end, orientation, strand], ...]
             end-start+1 == 34, because the CTCF ZF length is 34.
    '''
    temp_motifs = list()
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        fields = i.strip().split(',')
        # chromosome, start, end, orientation, strand
        temp_motifs.append([fields[0], int(fields[1]), int(fields[2]), fields[3], fields[4]])
    return temp_motifs

# 定义一个函数，输入一个序列的长度和一个区域列表，输出不在区域列表中的区域列表
def get_complement(chrom_size, region_list):
    # 初始化一个空列表
    complement = []
    # 初始化一个起始点
    start = 0
    # 遍历区域列表中的每个区域
    for region in region_list:
        # 如果区域的开始大于起始点，说明有一个不在区域列表中的区域
        if region[1] > start:
            end = region[1] - 1 # 结束点设为区域的开始减1
            complement.append([region[0], start, end, ".", "."]) # 把区域加入到列表中
        # 更新起始点为区域的结束加1
        print(region)
        start = region[2] + 1
    # 如果起始点小于等于序列的长度，说明还有一个不在区域列表中的区域
    if start <= chrom_size:
        end = chrom_size # 结束点设为序列的长度
        complement.append([region_list[-1][0], start, end, ".", "."]) # 把区域加入到列表中, non-motif的orientation和strand设置为"."
    # 返回区域列表
    return complement

@TimeStatistic
def find_non_motif_region(cell_line):
    '''
    @purpose: find non-motif region of motif in the cell line
    @anchor_type: 0: from CTCF motif in positive anchor;
                  1: from negative CTCF motif, i.e. the anchor but not positive anchor;
                  2: from non-motif position
    '''
    # * 正反motif的起点已全部格式化过，起点为0。而且start和end的点都包含在motif序列中。
    ## 1. load pos & neg motifs
    # pos motif
    pos_motifs = load_motifs(MOTIF_ON_ANCHOR_PATH.format(cell_line, cell_line))
    # neg motif
    neg_motifs = load_motifs(NEGATIVE_MOTIF_PATH.format(cell_line))
    
    ## 2. combine all motifs, get all CTCF motif positions
    # combine all motif to extract non-motif position
    all_motifs = pos_motifs + neg_motifs
    all_motifs = sorted(list(all_motifs), key=lambda x: (int(x[0][3:]), int(x[1])) if x[0][3:].isdigit() else (ord(x[0][3:]), int(x[1])))
    
    ## 3. extend CTCF motif region to safe & combine adjacent region
    # 在motif两旁(flanking)各延伸500bp，使得non-motif区域得到与motif安全的距离
    for i in range(len(all_motifs)):
        all_motifs[i][1] -= 500
        all_motifs[i][2] += 500
    # 将距离≤1000的motif region 连接成一个，用于筛选non-motif region
    all_motifs_region = defaultdict(list)
    i = 0
    combination_num = 0
    region_num = 0
    while i < len(all_motifs):
        temp_motif_region = [all_motifs[i][0], all_motifs[i][1], all_motifs[i][2]]
        i += 1
        while i < len(all_motifs) and all_motifs[i][0] == temp_motif_region[0] and all_motifs[i][1] - temp_motif_region[2] < 1001:
            temp_motif_region[2] = all_motifs[i][2]
            i += 1
            combination_num += 1
        all_motifs_region[temp_motif_region[0]].append(temp_motif_region)
        region_num += 1
    print(f"all_motifs num: {len(all_motifs)}, after {combination_num} combination, generate all_motifs_region: {region_num}")
    
    ## 4. get the non-motif regions that dont contain CTCF motif
    chrom_sizes = dict()
    non_motif_regions = list()
    with open(CHROM_SIZES, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields[0]) < 6:
                chrom_sizes[fields[0]] = int(fields[1])
    for chr in all_motifs_region:
        non_motif_regions.extend(get_complement(chrom_sizes[chr]-1, all_motifs_region[chr]))
    print(f"get non-motif regions, total non-motif regions: {len(non_motif_regions)}, containing {sum([1 for i in non_motif_regions if i[2]-i[1]+1<1000])} region which len < 1000")
    with open(NON_MOTIF_PATH.format(cell_line), 'w') as f:
        for non_motif in non_motif_regions:
            f.write(",".join(list(map(str, non_motif)))+'\n')

class NegLoopGenerator:
    def __init__(self, cell_line_list, neg_loop_type_list, quantity):
        '''
        @purpose: generate negative samples
        @param cell_line_list: list of required cell lines
        @param neg_loop_type_list: list of required loop types
        @param quantity: required number of each loop type
        '''
        assert type(neg_loop_type_list) in [range, list]
        assert set(neg_loop_type_list).issubset(range(6))
        self.FUNC_DICT = {0: self.__gen_neg_loop_type_0, 1: self.__gen_neg_loop_type_1, 2: self.__gen_neg_loop_type_2, 3: self.__gen_neg_loop_type_3, 4: self.__gen_neg_loop_type_4, 5: self.__gen_neg_loop_type_5}
        random.seed(RANDOM_SEED)
        self.CHROMOSOME_DICT = load_UCSC_hg38()
        self.LOOP_SIZE = [5000, 500000]

        self.cell_line_list = cell_line_list
        self.current_cell_line = None
        self.neg_loop_type_list = neg_loop_type_list
        self.quantity = quantity

    def gen_neg_sample(self):
        '''
        @purpose: generate negative samples
        @Input: 
            cell_line: cell line name
            neg_loop_type: 要生成的负样本的环的类型，包含六种。
        @params:
            anchor_type: 0: from CTCF motif in positive anchor;
                        1: from negative CTCF motif, i.e. the anchor but not positive anchor;
                        2: from non-motif position
            neg_loop_type: neg_loop_type: (anchor1_type, anchor2_type)
                        0: (0, 0), 1: (0, 1), 2: (0, 2),
                        3: (1, 1), 4: (1, 2), 5: (2, 2)
        '''
        start_time = time.perf_counter()
        for cell_line in self.cell_line_list:
            self.current_cell_line = cell_line
            print(f"cell_line: {self.current_cell_line}")
            for i in self.neg_loop_type_list:
                print(f"\tneg_loop_type: {i}")
                self.FUNC_DICT[i]()
        end_time = time.perf_counter()
        print(f"runtime: {end_time-start_time}")

    def __gen_neg_loop_type_0(self):
        '''
        @purpose: generate loop with pos motif anchor & pos motif anchor
        @output: self.in_len_range_loops & self.out_len_range_loops
        '''
        LOOP_TYPE = 0
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        pos_motifs = load_motifs(MOTIF_ON_ANCHOR_PATH.format(self.current_cell_line, self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1 两个pos motif有可能组成 pos loop，所以把其中一个序列打乱
            motif_1 = copy.deepcopy(pos_motifs[random.randint(0, len(pos_motifs)-1)])
            motif_1[3], motif_1[4] = ".", "."
            motif_1 = [motif_1[0], motif_1[1], motif_1[2], '.,.']
            seq_1 = shuffle_a_sequence(self.CHROMOSOME_DICT[motif_1[0]][motif_1[1]-483: motif_1[2]+484]) # flank to 1000bp
            motif_1.extend([get_mid_point(motif_1[1], motif_1[2]), seq_1])
            # motif 2
            motif_2 = self.__extend_anchor_features(pos_motifs[random.randint(0, len(pos_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __gen_neg_loop_type_1(self):
        '''
        @purpose: generate loop with pos motif anchor & neg motif anchor
        '''
        LOOP_TYPE = 1
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        pos_motifs = load_motifs(MOTIF_ON_ANCHOR_PATH.format(self.current_cell_line, self.current_cell_line))
        neg_motifs = load_motifs(NEGATIVE_MOTIF_PATH.format(self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1
            motif_1 = self.__extend_anchor_features(pos_motifs[random.randint(0, len(pos_motifs)-1)])
            # motif 2
            motif_2 = self.__extend_anchor_features(neg_motifs[random.randint(0, len(neg_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __gen_neg_loop_type_2(self):
        '''
        @purpose: generate loop with pos motif anchor & non-motif anchor
        '''
        LOOP_TYPE = 2
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        pos_motifs = load_motifs(MOTIF_ON_ANCHOR_PATH.format(self.current_cell_line, self.current_cell_line))
        non_motifs = load_motifs(NON_MOTIF_PATH.format(self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1
            motif_1 = self.__extend_anchor_features(pos_motifs[random.randint(0, len(pos_motifs)-1)])
            # motif 2
            motif_2 = self.__extend_non_motif_anchor_features(non_motifs[random.randint(0, len(non_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __gen_neg_loop_type_3(self):
        '''
        @purpose: generate loop with neg motif anchor & neg motif anchor
        '''
        LOOP_TYPE = 3
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        neg_motifs = load_motifs(NEGATIVE_MOTIF_PATH.format(self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1
            motif_1 = self.__extend_anchor_features(neg_motifs[random.randint(0, len(neg_motifs)-1)])
            # motif 2
            motif_2 = self.__extend_anchor_features(neg_motifs[random.randint(0, len(neg_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __gen_neg_loop_type_4(self):
        '''
        @purpose: generate loop with neg motif anchor & non-motif anchor
        '''
        LOOP_TYPE = 4
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        neg_motifs = load_motifs(NEGATIVE_MOTIF_PATH.format(self.current_cell_line, self.current_cell_line))
        non_motifs = load_motifs(NON_MOTIF_PATH.format(self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1
            motif_1 = self.__extend_anchor_features(neg_motifs[random.randint(0, len(neg_motifs)-1)])
            # motif 2
            motif_2 = self.__extend_non_motif_anchor_features(non_motifs[random.randint(0, len(non_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __gen_neg_loop_type_5(self):
        '''
        @purpose: generate loop with non-motif anchor & non-motif anchor
        '''
        LOOP_TYPE = 5
        self.in_len_range_loops = list()
        self.out_len_range_loops = list()
        non_motifs = load_motifs(NON_MOTIF_PATH.format(self.current_cell_line))
        # chromosome, start, end, orientation, strand 此处start和end都是从0算起，并且序列包含start和end位置的碱基
        while len(self.in_len_range_loops) < self.quantity or len(self.out_len_range_loops) < self.quantity:
            # motif 1
            motif_1 = self.__extend_non_motif_anchor_features(non_motifs[random.randint(0, len(non_motifs)-1)])
            # motif 2
            motif_2 = self.__extend_non_motif_anchor_features(non_motifs[random.randint(0, len(non_motifs)-1)])
            self.__binning(motif_1, motif_2)
        # output format
        # chr1,start,end,orientation,strand,mid_point,seq_1,chr2,start,end,orientation,strand,mid_point,seq_2
        # |-----------------------chr1---------------------||------------------------chr2-------------------|
        self.print_neg_loop(loop_type=LOOP_TYPE)

    def __extend_anchor_features(self, motif):
        '''
        add midpoint & 1000bp seq to pos & neg anchor features
        '''
        motif = copy.deepcopy(motif)
        motif = [motif[0], motif[1], motif[2], motif[3]+','+motif[4]]
        seq = self.CHROMOSOME_DICT[motif[0]][motif[1]-483: motif[2]+484]
        motif.extend([get_mid_point(motif[1], motif[2]), seq])
        return motif

    def __extend_non_motif_anchor_features(self, motif):
        '''
        add midpoint & 1000bp seq to non-motif anchor features
        '''
        motif = copy.deepcopy(motif)
        motif = [motif[0], motif[1], motif[2], motif[3]+','+motif[4]]
        # change motif_2 start&end to 34bp to get a 1000bp sequence at follow
        motif[1] = random.randint(motif[1]+483, motif[2]-516)
        motif[2] = motif[1]+33
        seq = self.CHROMOSOME_DICT[motif[0]][motif[1]-483: motif[2]+484]
        motif.extend([get_mid_point(motif[1], motif[2]), seq])
        return motif

    def __binning(self, region_1, region_2):
        """
        binning the loops
        """
        if region_1[0] == region_2[0]: # must locate on same chr
            if region_1[1] > region_2[1]:
                region_1, region_2 = region_2, region_1
            loop_length = region_2[1] - region_1[1]
            if loop_length < self.LOOP_SIZE[0]:
                return
            if self.LOOP_SIZE[0] <= loop_length <= self.LOOP_SIZE[1] and len(self.in_len_range_loops) < self.quantity:
                self.in_len_range_loops.append([region_1, region_2])
            elif self.LOOP_SIZE[1] < loop_length and len(self.out_len_range_loops) < self.quantity:
                self.out_len_range_loops.append([region_1, region_2])

    def print_neg_loop(self, loop_type):
        with open(NEG_LOOP_PATH.format(self.current_cell_line, loop_type, 0), 'w') as f:
            for i in self.in_len_range_loops:
                f.write('\t'.join(map(str, i[0]+i[1])) + '\n')
        with open(NEG_LOOP_PATH.format(self.current_cell_line, loop_type, 1), 'w') as f:
            for i in self.out_len_range_loops:
                f.write('\t'.join(map(str, i[0]+i[1])) + '\n')


if __name__ == '__main__':
    # bedpe_accession = "LHG0066V" # LHG0066V LHG0052H
    # cell_line = "GM12878"
    # cell_line = "K562"
    # cell_line = "RWPE1"


    # filter_negative_motif(cell_line='GM12878')
    # for cell_line in CELL_LINE_LIST:
    #     print(f"[INFO] Preprocessing {cell_line} loop")
    #     filter_negative_motif(cell_line)

    # find_non_motif_region(cell_line='GM12878')
    # for cell_line in CELL_LINE_LIST:
    #     print(f"[INFO] Preprocessing {cell_line} loop")
    #     find_non_motif_region(cell_line)

    nsg = NegLoopGenerator(cell_line_list=CELL_LINE_LIST, neg_loop_type_list=range(6), quantity=1500)
    nsg.gen_neg_sample()



