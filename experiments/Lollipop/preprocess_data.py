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
@author: sjlgg
'''


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from source.parameters import POS_LOOP_PATH, NEG_LOOP_PATH, ROOT_DIR, CELL_LINE_LIST

LOLLIPOP_PATH = ROOT_DIR+"/experiments/Lollipop/"

def preprocess_pos_loop(cell_line_list):
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        out_file = open(LOLLIPOP_PATH+"/training_data/{}.pos_loop".format(cell_line), 'w')
        out_file.write("chrom\tstart1\tstart2\tresponse\tlength\tmotif_pattern\tavg_motif_strength\tstd_motif_strength\tavg_conservation\tstd_conservation\tavg_HS\tstd_HS\tHS_in-between\tHS_left\tHS_right\n")
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
                out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\n")
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
                    out_file.write(f"{chrom}\t{start1}\t{start2}\t{response}\t{length}\t{motif_pattern}\t{avg_motif_strength}\t{std_motif_strength}\t{avg_conservation}\t{std_conservation}\t{avg_HS}\t{std_HS}\t{HS_in_between}\t{HS_left}\t{HS_right}\n")
        out_file.close()

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


if __name__ == "__main__":
    preprocess_pos_loop(["GM12878"])

