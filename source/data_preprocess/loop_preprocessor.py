"""
@author: 孙嘉良
@purpose: check & preprocess loop data to gain positive samples
"""
import sys, os, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import *
from Bio import SeqIO
# from utils import TimeStatistic

def scan_CTCF_motif_on_chromosome():
    '''
    Scan through CTCF motifs and return position and string of motif if found
    此函数用于检查一下ENCODE下的bedpe能行不
    '''
    chia_pet_path = BEDPE_PATH+r'/LHG0066V.e500.clusters.cis.BE3'
    chia_pet_file = open(chia_pet_path, 'r')
    chia_pet_lines = chia_pet_file.readlines()
    chia_pet_file.close()

    ZINC_FINGER = False
    for line in chia_pet_lines:
        loop_anchor_pair_info = line.strip().split()
        if loop_anchor_pair_info[0] == loop_anchor_pair_info[3]:
            chr = loop_anchor_pair_info[0]
            chr_path = ROOT_DIR+r"/data/GCF_000001405.40/"+CHROMOSOME_DICT[chr]+r'.fasta'
            seq_record = SeqIO.read(chr_path, "fasta")
            # print(f"{seq_record.id}, {repr(seq_record.seq)}, {len(seq_record)}")
        else:
            print('chr1 and chr2 not same')
            raise Exception("chr1 and chr2 not same")
        start1, end1, start2, end2 = int(loop_anchor_pair_info[1]), int(loop_anchor_pair_info[2]), int(loop_anchor_pair_info[4]), int(loop_anchor_pair_info[5])
        anchor1, anchor2 = seq_record.seq[start1:end1+1], seq_record.seq[start2:end2+1]
        print(f'anchor1: {len(anchor1)}, anchor2: {len(anchor2)}')
        similarity_list = scan_CTCF_motif_on_anchor(anchor=anchor1, zinc_finger=ZINC_FINGER)
        print(similarity_list[:3])
        similarity_list = scan_CTCF_motif_on_anchor(anchor=anchor1, reverse=True, zinc_finger=ZINC_FINGER)
        print(similarity_list[:3])
        similarity_list = scan_CTCF_motif_on_anchor(anchor=anchor2, zinc_finger=ZINC_FINGER)
        print(similarity_list[:3])
        similarity_list = scan_CTCF_motif_on_anchor(anchor=anchor2, reverse=True, zinc_finger=ZINC_FINGER)
        print(similarity_list[:3])
        exit()
    '''
    对于/CHASS/data/ChIA-PET/CTCF/bedpe/LHG0066V.e500.clusters.cis.BE3的第一行
    chr10	71711	72318	chr10	158517	159047	3
    anchor1上的430处的 正向的motif [8.746880999999998, 430, Seq('TTGAACCCAGGAGGCAGAG')] 
             和415处的 正向的锌指motif [12.177358999999997, 415, Seq('AGGTAGGAGAATTGCTTGAACCCAGGAGGCAGAG')]
             的CTCF motif的序列相似度都很高
    anchor2上的87处的 反向的motif [9.212357, 87, Seq('GAGACGGGGTTTCACCATG')]
             和87处的 反向的锌指motif [13.563111000000003, 87, Seq('GAGACGGGGTTTCACCATGTTGGCCAGGCTGGTC')]
             的序列相似度都很高
    '''

def scan_CTCF_motif_on_anchor(anchor, reverse=False, zinc_finger=False):
    '''
    附属于scan_CTCF_motif_on_chromosome，用于在anchor序列上扫描CTCF_motif，并计算对应的序列相似度，并记录位置
    '''
    assert type(reverse) == bool and type(zinc_finger) == bool
    similarity_list = list()
    if reverse == False:
        if zinc_finger == False:
            ctcf_pwm = CTCF_PWM
        else:
            ctcf_pwm = CTCF_ZF_PWM
    else:
        if zinc_finger == False:
            ctcf_pwm = CTCF_PWM_REVERSE
        else:
            ctcf_pwm = CTCF_ZF_PWM_REVERSE
    for i in range(len(anchor)-len(ctcf_pwm)+1):
        similarity_score = 0
        for j in range(len(ctcf_pwm)):
            # print(j, anchor[i+j], CTCF_PWM_INDEX[anchor[i+j]], ctcf_pwm[j][CTCF_PWM_INDEX[anchor[i+j]]])
            similarity_score += math.log2(ctcf_pwm[j][CTCF_PWM_INDEX[anchor[i+j]]] / 0.25)
        similarity_list.append([similarity_score, i, anchor[i:i+len(ctcf_pwm)]])
    return sorted(similarity_list, reverse=True)

def load_UCSC_hg38():
    # UCSC hg38
    local_chromosome_dict = CHROMOSOME_DICT.copy()
    for seq_record in SeqIO.parse(ROOT_DIR+r"/data/hg38/hg38.fa", "fasta"):
        if seq_record.id in local_chromosome_dict.keys():
            local_chromosome_dict[seq_record.id] = seq_record.seq.upper()
    return local_chromosome_dict

def load_NCBI_GRCh38_p14():
    # NCBI GCF_000001405.40（GRCh38.p14）
    local_chromosome_dict = CHROMOSOME_DICT.copy()
    for chr in CHROMOSOME_DICT:
        chr_path = ROOT_DIR+r"/data/GCF_000001405.40/"+CHROMOSOME_DICT[chr]+r'.fasta'
        seq_record = SeqIO.read(chr_path, "fasta")
        local_chromosome_dict[chr] = seq_record.seq.upper()
    return local_chromosome_dict

# @TimeStatistic
def check_fimo_hg38_CTCF_motif_on_new_hg38():
    '''
    原目的：检查用FIMO根据MA1929.1 motif在FIMO在线hg38中扫描到的motif位置是否与本人新下的hg38符合。结果：同时符合两个版本（UCSC, NCBI）的hg38的CTCF motif。
    新目的：过滤不在染色体上的FIMO CTCF motif记录，比如有些motif在chr12_GL877875v1_alt上。
    '''
    f = open(FIMO_RESULT_PATH+r"/FIMO.tsv", 'r')
    fimo_lines = f.readlines()
    f.close()
    title = fimo_lines[0]
    fimo_lines = fimo_lines[1:]

    ## 在UCSC里下的2020/3/17版本的hg38中，我自己将FIMO的结果和hg38比对了一下是吻合的，正义链、反义链、motif位置都对。
    local_chromosome_dict = load_UCSC_hg38()
    ## 在NCBI里下的GCF_000001405.40（GRCh38.p14）版本的hg38中，我自己将FIMO的结果和hg38比对了一下是吻合的，正义链、反义链、motif位置都对。
    # local_chromosome_dict = load_NCBI_GRCh38_p14()

    unfitted_record = dict()
    for k in local_chromosome_dict:
        unfitted_record[k] = 0

    # print(local_chromosome_dict)

    f = open(FIMO_RESULT_PATH+r"/FIMO_filtered.tsv", 'w')
    f.write(title)

    for i, l in enumerate(fimo_lines):
        if i % 10000 == 0:
            print(i)
        if l.strip() == "":
            break
        items = l.strip().split()
        items[9] = items[9].upper()
        chr, start, end, strand, matched_sequence = items[2], int(items[3]), int(items[4]), items[5], items[9]
        if chr not in CHROMOSOME_DICT.keys(): continue
        assert strand == "+" or strand == "-"

        seq = local_chromosome_dict[chr]
        if strand == "+" and seq[start-1:end] != matched_sequence:
            print(f"{i}, {chr}, {start}, {end}, {strand}, {matched_sequence}, ?{seq[start-1:end]}")
            unfitted_record[chr] += 1
        elif strand == "-" and seq[start-1:end].reverse_complement() != matched_sequence:
            # 反义链的start和end是从3’开始到5‘，正义链的start和end是从5‘开始到3’
            print(f"{i}, {chr}, {start}, {end}, {strand}, {matched_sequence}, ?{seq[start-1:end].reverse_complement()}")
            unfitted_record[chr] += 1
        else:
            f.write("\t".join(items)+"\n")
    f.close()
    print(f"unfitted_record: {unfitted_record}")

# @TimeStatistic
def sort_FIMO_result():
    f = open(FIMO_RESULT_PATH+r"/FIMO_filtered.tsv", 'r')
    lines = f.readlines()
    f.close()
    title = lines[0]
    lines = [l.strip().split() for l in lines[1:]]
    for l in lines:
        l[2] = l[2][3:]
        if l[2] == "X":
            l[2] = 23
        elif l[2] == "Y":
            l[2] = 24
        else:
            l[2] = int(l[2])
    lines.sort(key=lambda x: (x[2], x[5], int(x[3]), int(x[4])))
    results = list()
    for l in lines:
        if l[2] == 23:
            l[2] = "chrX"
        elif l[2] == 24:
            l[2] = "chrY"
        else:
            l[2] = "chr"+str(l[2])
        results.append("\t".join(l)+'\n')

    f = open(FIMO_RESULT_PATH+r"/FIMO_filtered_sorted.tsv", 'w')
    f.write(title)
    f.writelines(results)
    f.close()

def load_CTCF_motifs(fimo_file_path=""):
    '''
    读取CTCF motifs的筛选后、排序后的文件——FIMO_filtered_sorted.tsv
    output: first key = 以'chr'为前缀的键, second key = strand (+/-), value = list of [start, end]
    '''
    if fimo_file_path == "":
        fimo_file_path = FIMO_CUSTOMISED_RESULT_PATH
    f = open(fimo_file_path.format(""), 'r')
    lines = f.readlines()[1:]
    f.close()

    CTCF_motifs = dict()
    for l in lines:
        items = l.strip().split()
        if items[2] not in CTCF_motifs.keys():
            CTCF_motifs[items[2]] = {"+":list(), "-":list()}
        CTCF_motifs[items[2]][items[5]].append([int(items[3]), int(items[4]), "f"])
    
    f = open(fimo_file_path.format("_reverse"), 'r')
    lines = f.readlines()[1:]
    f.close()

    for l in lines:
        items = l.strip().split()
        if items[2] not in CTCF_motifs.keys():
            CTCF_motifs[items[2]] = {"+":list(), "-":list()}
        CTCF_motifs[items[2]][items[5]].append([int(items[3]), int(items[4]), "r"])

    for chr in CTCF_motifs:
        for j in ["+", "-"]:
            CTCF_motifs[chr][j].sort()
    return CTCF_motifs

# @TimeStatistic
def filter_loop_containing_CTCF(bedpe_path=BEDPE_PATH, bedpe_filtered_path=BEDPE_FILTERED_PATH, motif_on_anchor_path=MOTIF_ON_ANCHOR_PATH):
    '''
    从ChIA-PET的bedpe文件里，筛选出包含CTCF motif的loop
    bedpe似乎单纯是记录环的位置，所以不考虑anchor所在链的正反，可能都是从正链的5'（即反链的3'）算起的。
    bedpe里的start和end是，视第一个碱基位置为0
    从check_fimo_hg38_CTCF_motif_on_new_hg38的情况来看，FIMO结果里的start和end应该是视第一个碱基位置为1
    '''
    chia_pet_path = bedpe_path
    chia_pet_file = open(chia_pet_path, 'r')
    chia_pet_lines = chia_pet_file.readlines()
    chia_pet_file.close()

    # local_chromosome_dict = load_NCBI_GRCh38_p14() # not used
    CTCF_motifs = load_CTCF_motifs()

    f_out = open(bedpe_filtered_path, 'w')
    all_anchor_motifs_set = set()
    for i, l in enumerate(chia_pet_lines):
        if i % 20000 == 0:
            print(i)
        loop_anchor_pair_info = l.strip().split()
        if loop_anchor_pair_info[0] == loop_anchor_pair_info[3]:
            chr = loop_anchor_pair_info[0]
            if chr == "chrM" or "_" in chr:
                continue
        else:
            print('chr1 and chr2 not same')
            raise Exception("chr1 and chr2 not same")
        start1, end1, start2, end2 = map(int, loop_anchor_pair_info[1:3]+loop_anchor_pair_info[4:6])
        anchor1_motifs_list, anchor2_motifs_list = list(), list() # store the start and end of ctcf motifs on the anchor
        # scan ctcf motif on 2 anchors
        for anchor_motifs_list, start, end in [[anchor1_motifs_list, start1, end1], [anchor2_motifs_list, start2, end2]]:
            for strand in ["+", "-"]:
                for ctcf_start, ctcf_end, orientation in CTCF_motifs[chr][strand]:
                    formated_ctcf_start, formated_ctcf_end = ctcf_start-1, ctcf_end-1 # 将CTCF motifs的start和end格式化为从0开始
                    if start <= formated_ctcf_start and formated_ctcf_end <= end:
                        anchor_motifs_list.append(f"{formated_ctcf_start},{formated_ctcf_end},{orientation},{strand}")
                        all_anchor_motifs_set.add(f"{chr},{formated_ctcf_start},{formated_ctcf_end},{orientation},{strand}")
                    elif end < formated_ctcf_start:
                        break
        anchor1_motifs_str, anchor2_motifs_str = ";".join(anchor1_motifs_list), ";".join(anchor2_motifs_list)
        # // loop_info_with_motifs = "\t".join(loop_anchor_pair_info[0:6])+"\t"+anchor1_motifs_str+"\t"+anchor2_motifs_str
        # // if anchor1_motifs_str != '' and anchor2_motifs_str != '':
        # //     f_out.write(loop_info_with_motifs+"\n")
        if anchor1_motifs_str == "": anchor1_motifs_str = "."
        if anchor2_motifs_str == "": anchor2_motifs_str = "."
        loop_info_with_motifs = "\t".join(loop_anchor_pair_info[0:13])+"\t"+anchor1_motifs_str+"\t"+anchor2_motifs_str
        f_out.write(loop_info_with_motifs+"\n")
    f_out.close()
    # * 此处输出的pos_motif，包含single类型的环(即仅单个anchor上包含CTCF motif)上的motif。也包含convergent, tandem, divergent上两个anchor上的motif
    # * 由于在此函数前执行了filter_MICC_by_PET_and_FDR，所以此处的pos_motif都是经过这个函数过滤的高质量环上的。
    all_anchor_motifs_list = sorted(list(all_anchor_motifs_set), key=lambda x: (int(x.split(',')[0][3:]), int(x.split(',')[1])) if x.split(',')[0][3:].isdigit() else (ord(x.split(',')[0][3:]), int(x.split(',')[1])))
    motif_out = open(motif_on_anchor_path, 'w')
    for i in all_anchor_motifs_list:
        motif_out.write(i+"\n")
    motif_out.close()

# @TimeStatistic
def count_loop_type(bedpe_filtered_path=BEDPE_FILTERED_PATH):
    """
    计算各种loop类型的数量：
    convergent, tandem, divergent, single, none是loop的两个anchors的CTCF-binding site在不同的motif方向（motif orientation）下的五种情况
    由于现在我只要两个anchors都有CTCF motif的环，所以目前有四类环，既可以convergent也可以tandem，convergent，tandem，divergent，others。
    设：convergent:1 tandem:2 divergent:3 single:4 none:5 两种类型都可能的就连续，如既可以convergent也可以tandem:12
    """
    counter_dict = {'1':0, '2':0, '3':0, '12':0, '13':0, '23':0, '123':0, '4':0, '5':0}
    convergent_anchor_pair = [['f,+','r,+'], ['f,+','f,-'], ['r,-','r,+'], ['r,-','f,-']]
    tandem_anchor_pair = [['f,+','f,+'], ['f,+','r,-'], ['r,+','r,+'], ['r,+','f,-'], ['f,-','r,+'], ['f,-','f,-'], ['r,-','f,+'], ['r,-','r,-']]
    divergent_anchor_pair = [['r,+','f,+'], ['r,+','r,-'], ['f,-','f,+'], ['f,-','r,-']]

    f = open(bedpe_filtered_path, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        anchor1_motifs_str, anchor2_motifs_str = line.strip().split()[13:15]
        if anchor1_motifs_str == '.' and anchor2_motifs_str == '.':
            counter_dict['5'] += 1
            continue
        elif anchor1_motifs_str == '.' or anchor2_motifs_str == '.':
            counter_dict['4'] += 1
            continue
        anchor1_motif_orientations = [",".join(p.split(",")[2:]) for p in anchor1_motifs_str.split(";")]
        anchor2_motif_orientations = [",".join(p.split(",")[2:]) for p in anchor2_motifs_str.split(";")]
        type_sign = set()
        for orientation1 in anchor1_motif_orientations:
            for orientation2 in anchor2_motif_orientations:
                anchor_pair = [orientation1, orientation2]
                if anchor_pair in convergent_anchor_pair:
                    type_sign.add(1)
                elif anchor_pair in tandem_anchor_pair:
                    type_sign.add(2)
                elif anchor_pair in divergent_anchor_pair:
                    type_sign.add(3)
        type_sign = "".join(map(str, sorted(list(type_sign))))
        counter_dict[type_sign] += 1
    print(counter_dict)
    return counter_dict

# @TimeStatistic
def filter_MICC_by_PET_and_FDR():
    '''
    #PET and FDR are two parameters in ChIA-PET2 result which end with '.MICC'
    Lollipop对ChIA-PET2结果的过滤条件：false discovery rate (FDR) ≤ 0.05 and paired-end tag (PET) number ≥ 2
    input: GM12878.interactions.MICC
    output: GM12878.significant.interactions.MICC
    '''
    f = open(chia_pet2_result_path+f"/{cell_line}.interactions.MICC", "r")
    micc_lines = f.readlines()
    f.close()
    micc_lines = micc_lines[1:]

    new_micc_lines = list()
    for i in micc_lines:
        items = i.strip().split()
        if "_" in items[0] or "_" in items[3]: continue
        if float(items[10]) >= 2 and float(items[12]) <= 0.05:
            new_micc_lines.append(i)
    new_micc_lines = sorted(list(new_micc_lines), key=lambda x: (int(x.split()[0][3:]), int(x.split()[1]), int(x.split()[4])) if x.split()[0][3:].isdigit() else (ord(x.split()[0][3:]), int(x.split()[1]), int(x.split()[4])))
    out = open(chia_pet2_result_path+f"/{cell_line}.significant.interactions.MICC", "w")
    out.writelines(new_micc_lines)
    out.close()

# @TimeStatistic
def split_intra_inter_from_MICC():
    '''
    split MICC into (1) intra & (2) inter
    ChIA-PET2在生成MICC的时候把intra和inter合在一起了，我把结果分开
    input: GM12878.significant.interactions.MICC
    output: GM12878.significant.interactions.intra.MICC & GM12878.significant.interactions.inter.MICC
    '''
    f = open(chia_pet2_result_path+f"/{cell_line}.significant.interactions.MICC", "r")
    micc_lines = f.readlines()
    f.close()

    # fields = micc_lines[0]
    # micc_lines = micc_lines[1:]

    intra_out = open(chia_pet2_result_path+f"/{cell_line}.significant.interactions.intra.MICC", "w")
    # intra_out.write(fields)
    inter_out = open(chia_pet2_result_path+f"/{cell_line}.significant.interactions.inter.MICC", "w")
    # inter_out.write(fields)
    for i in micc_lines:
        items = i.split()
        chr1, chr2 = items[0], items[3]
        if chr1 == chr2:
            intra_out.write(i)
        else:
            inter_out.write(i)

    intra_out.close()
    inter_out.close()


if __name__ == "__main__":
    # bedpe_accession = "LHG0052H" # LHG0066V LHG0052H
    # cell_line = "GM12878"
    # cell_line = "K562"
    # cell_line = "RWPE1"
    # cell_line = "Caco-2"


    # FIMO_RESULT_PATH = ROOT_DIR+r"/data/FIMO_RESULT_PATH/MA0139.1_reverse_hg38_FIMO_result/"+CURRENT_P_VALUE+"/"
    # FIMO_CUSTOMISED_RESULT_PATH = ROOT_DIR+"/data/FIMO_RESULT_PATH/MA0139.1{}_hg38_FIMO_result/"+CURRENT_P_VALUE+"/FIMO_filtered_sorted.tsv"
    # BEDPE_FILTERED_PATH = ROOT_DIR+r'/data/ChIA-PET/CTCF/bedpe_filtered/MA0139.1/'+CURRENT_P_VALUE+"/"


    # BEDPE_PATH = BEDPE_PATH.format(bedpe_accession)
    # BEDPE_FILTERED_PATH = BEDPE_FILTERED_PATH.format(bedpe_accession)
    # MOTIF_ON_ANCHOR_PATH = MOTIF_ON_ANCHOR_PATH.format(bedpe_accession)
    # chia_pet2_result_path = CHIA_PET2_RESULT_PATH.format(cell_line)

    # scan_CTCF_motif_on_chromosome()

    ## 过滤不在染色体上的motif
    # check_fimo_hg38_CTCF_motif_on_new_hg38()
    # sort_FIMO_result()

    # motifs = load_CTCF_motifs()

    # * 下面四个函数是对ChIA-PET2的结果中的MICC文件进行处理，
    # * (1)过滤低可信度MICC结果;(2)从MICC中分离染色体内loop;(3)标注两个anchor上的CTCF基序的链和方向;(4)统计环的类型数量
    # filter_MICC_by_PET_and_FDR()
    # split_intra_inter_from_MICC()

    # filter_loop_containing_CTCF(bedpe_path=chia_pet2_result_path+'/{}.significant.interactions.intra.MICC'.format(cell_line), bedpe_filtered_path=chia_pet2_result_path+'/{}.significant.interactions.intra.filtered.MICC'.format(cell_line), motif_on_anchor_path=chia_pet2_result_path+'/{}.pos_motif'.format(cell_line))
    # count_loop_type(bedpe_filtered_path=chia_pet2_result_path+'/{}.significant.interactions.intra.filtered.MICC'.format(cell_line))

    # * 处理所有的cell line
    for cell_line in CELL_LINE_LIST:
        print(f"[INFO] Preprocessing {cell_line} loop")
        chia_pet2_result_path = CHIA_PET2_RESULT_PATH.format(cell_line)
        # filter_MICC_by_PET_and_FDR()
        # split_intra_inter_from_MICC()

        # filter_loop_containing_CTCF(bedpe_path=chia_pet2_result_path+'/{}.significant.interactions.intra.MICC'.format(cell_line), bedpe_filtered_path=chia_pet2_result_path+'/{}.significant.interactions.intra.filtered.MICC'.format(cell_line), motif_on_anchor_path=chia_pet2_result_path+'/{}.pos_motif'.format(cell_line))
        count_loop_type(bedpe_filtered_path=chia_pet2_result_path+'/{}.significant.interactions.intra.filtered.MICC'.format(cell_line))

