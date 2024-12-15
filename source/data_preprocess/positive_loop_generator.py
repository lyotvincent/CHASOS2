'''
@author: 孙嘉良
@purpose: generate positive loop samples
'''


import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from parameters import *
from loop_preprocessor import load_UCSC_hg38, count_loop_type
from utils import get_mid_point

class PositiveLoopGenerator:
    def __init__(self, cell_line_list, quantity=None):
        '''
        @purpose: generate positive samples
        @param cell_line_list: list of required cell lines
        @param quantity: required number of each cell line
        '''
        self.cell_line_list = cell_line_list
        self.CHROMOSOME_DICT = load_UCSC_hg38()
        self.quantity = quantity
        random.seed(RANDOM_SEED)
        
    def gen_pos_sample(self):
        start_time = time.perf_counter()
        for cell_line in self.cell_line_list:
            self.current_cell_line = cell_line
            print(f"cell_line: {self.current_cell_line}")
            self.__get_cell_line_quantity()
            self.__gen_pos_sample_for_a_cell_line()
        end_time = time.perf_counter()
        print(f"runtime: {end_time-start_time}")

    def __get_cell_line_quantity(self):
        self.quantity = count_loop_type(bedpe_filtered_path=(CHIA_PET2_RESULT_PATH+'/{}.significant.interactions.intra.filtered.MICC').format(self.current_cell_line, self.current_cell_line))["1"]

    def __gen_pos_sample_for_a_cell_line(self):
        micc_file_path = (CHIA_PET2_RESULT_PATH+'/{}.significant.interactions.intra.filtered.MICC').format(self.current_cell_line, self.current_cell_line)
        loops_type_1 = list() # 22709/53741     42.256%     64.4%
        loops_type_2 = list() # 11674/53741     21.7227%    32.2%
        loops_type_3 = list() # 847/53741       1.576%      1.5%
        loops_type_4 = list() # 10952/53741     20.379%     1.0%
        loops_type_5 = list() # 492/53741       0.9155%     0.9%
        LOOP_TYPE_RATIO = [0.644, 0.322, 0.015, 0.01, 0.009]
        f = open(micc_file_path, 'r')
        lines = f.readlines()
        f.close()
        random.shuffle(lines)
        while len(lines) > 0 and\
             (len(loops_type_1) < self.quantity*LOOP_TYPE_RATIO[0] or\
              len(loops_type_2) < self.quantity*LOOP_TYPE_RATIO[1] or\
              len(loops_type_3) < self.quantity*LOOP_TYPE_RATIO[2] or\
              len(loops_type_4) < self.quantity*LOOP_TYPE_RATIO[3] or\
              len(loops_type_5) < self.quantity*LOOP_TYPE_RATIO[4]):
            line = lines.pop()
            fields = line.strip().split()
            loop_type = self.__judge_loop_type(fields[13], fields[14])
            if loop_type == False: continue
            # anchor_1 format: chr1,start,end,orientation,strand,mid_point,seq_1
            if loop_type in [1, 2, 3]:
                anchor_1 = self.__extend_anchor_features(fields[0], fields[13])
                anchor_2 = self.__extend_anchor_features(fields[3], fields[14])
            elif loop_type in [4, 5]:
                anchor_1, anchor_2 = self.__extend_anchor_features_for_non_motif(fields)
            a_loop = anchor_1+anchor_2
            if loop_type == 1 and len(loops_type_1) < self.quantity*LOOP_TYPE_RATIO[0]:
                loops_type_1.append(a_loop)
            elif loop_type == 2 and len(loops_type_2) < self.quantity*LOOP_TYPE_RATIO[1]:
                loops_type_2.append(a_loop)
            elif loop_type == 3 and len(loops_type_3) < self.quantity*LOOP_TYPE_RATIO[2]:
                loops_type_3.append(a_loop)
            elif loop_type == 4 and len(loops_type_4) < self.quantity*LOOP_TYPE_RATIO[3]:
                loops_type_4.append(a_loop)
            elif loop_type == 5 and len(loops_type_5) < self.quantity*LOOP_TYPE_RATIO[4]:
                loops_type_5.append(a_loop)
        with open(POS_LOOP_PATH.format(self.current_cell_line, 1), 'w') as f:
            for i in loops_type_1:
                f.write('\t'.join(map(str, i)) + '\n')
        with open(POS_LOOP_PATH.format(self.current_cell_line, 2), 'w') as f:
            for i in loops_type_2:
                f.write('\t'.join(map(str, i)) + '\n')
        with open(POS_LOOP_PATH.format(self.current_cell_line, 3), 'w') as f:
            for i in loops_type_3:
                f.write('\t'.join(map(str, i)) + '\n')
        with open(POS_LOOP_PATH.format(self.current_cell_line, 4), 'w') as f:
            for i in loops_type_4:
                f.write('\t'.join(map(str, i)) + '\n')
        with open(POS_LOOP_PATH.format(self.current_cell_line, 5), 'w') as f:
            for i in loops_type_5:
                f.write('\t'.join(map(str, i)) + '\n')

    def __judge_loop_type(self, anchor1_motifs_str, anchor2_motifs_str):
        '''
        @definition: (1) convergent, tandem, divergent, single, none是loop的两个anchors的CTCF-binding site
                        在不同的motif方向(motif orientation)下的五种情况。
                     (2) 设环类型: convergent:1 tandem:2 divergent:3 single:4 none:5 
                        两种类型都可能的就连续，如既可以convergent也可以tandem:12
        '''
        CONVERGENT_ANCHOR_PAIR = [['f,+','r,+'], ['f,+','f,-'], ['r,-','r,+'], ['r,-','f,-']]
        TANDEM_ANCHOR_PAIR = [['f,+','f,+'], ['f,+','r,-'], ['r,+','r,+'], ['r,+','f,-'], ['f,-','r,+'], ['f,-','f,-'], ['r,-','f,+'], ['r,-','r,-']]
        DIVERGENT_ANCHOR_PAIR = [['r,+','f,+'], ['r,+','r,-'], ['f,-','f,+'], ['f,-','r,-']]
        if anchor1_motifs_str == '.' and anchor2_motifs_str == '.':
            return 5
        elif anchor1_motifs_str == '.' or anchor2_motifs_str == '.':
            return 4
        anchor1_motif_orientations = [",".join(p.split(",")[2:]) for p in anchor1_motifs_str.split(";")]
        anchor2_motif_orientations = [",".join(p.split(",")[2:]) for p in anchor2_motifs_str.split(";")]
        type_sign = set()
        for orientation1 in anchor1_motif_orientations:
            for orientation2 in anchor2_motif_orientations:
                anchor_pair = [orientation1, orientation2]
                if anchor_pair in CONVERGENT_ANCHOR_PAIR:
                    type_sign.add(1)
                elif anchor_pair in TANDEM_ANCHOR_PAIR:
                    type_sign.add(2)
                elif anchor_pair in DIVERGENT_ANCHOR_PAIR:
                    type_sign.add(3)
        type_sign = "".join(map(str, sorted(list(type_sign))))
        ABANDONED_LOOP_TYPE = ['12', '13', '23', '123']
        RESERVED_LOOP_TYPE = ['1', '2', '3', '4', '5']
        if type_sign in ABANDONED_LOOP_TYPE:
            return False
        elif type_sign in RESERVED_LOOP_TYPE:
            return int(type_sign)

    def __extend_anchor_features(self, chr, anchor_motif_orientations):
        '''
        add midpoint & 1000bp seq to anchor features
        '''
        motifs = anchor_motif_orientations.strip().split(';')
        motif = motifs[random.randint(0, len(motifs) - 1)]
        motif_start, motif_end, orientation, strand = motif.split(',')
        motif_start, motif_end = int(motif_start), int(motif_end)
        anchor = [chr, motif_start, motif_end, orientation+','+strand]
        seq = self.CHROMOSOME_DICT[chr][motif_start-483: motif_end+484]
        anchor.extend([get_mid_point(motif_start, motif_end), seq])
        return anchor

    def __extend_anchor_features_for_non_motif(self, fields):
        if fields[13] == '.':
            anchor_1_mid_point = round(get_mid_point(int(fields[1]), int(fields[2])))
            seq_1 = self.CHROMOSOME_DICT[fields[0]][anchor_1_mid_point-500: anchor_1_mid_point+500]
            anchor_1 = [fields[0], fields[1], fields[2], ".,.", anchor_1_mid_point, seq_1]
        else:
            anchor_1 = self.__extend_anchor_features(fields[0], fields[13])
        if fields[14] == '.':
            anchor_2_mid_point = round(get_mid_point(int(fields[4]), int(fields[5])))
            seq_2 = self.CHROMOSOME_DICT[fields[3]][anchor_2_mid_point-500: anchor_2_mid_point+500]
            anchor_2 = [fields[3], fields[4], fields[5], ".,.", anchor_2_mid_point, seq_2]
        else:
            anchor_2 = self.__extend_anchor_features(fields[3], fields[14])
        return anchor_1, anchor_2


if __name__ == '__main__':
    plg = PositiveLoopGenerator(cell_line_list=CELL_LINE_LIST)
    plg.gen_pos_sample()