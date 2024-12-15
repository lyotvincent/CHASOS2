
'''
    0   chrom
    1   start1
    2   start2
    3   respons
    4   length                  1
    5   motif_pattern           2
    6   avg_motif_strength      3
    7   std_motif_strength      4
    8   avg_conservation        5
    9   std_conservation        6
    10  avg_HS                  7
    11  std_HS                  8
    12  HS_in-between           9
    13  HS_left                 10
    14  HS_right                11
    15  avg_anchor_score        12
    16  std_anchor_score        13
    17  anchor_score1           14
    18  anchor_score2           15
    19  avg_ocr_score           16
    20  std_ocr_score           17
    21  ocr_score1              18
    22  ocr_score2              19
'''

import sys, os, random, sqlite3
import joblib
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from source.parameters import PRETRAINED_MODEL_PATH, FINE_TUNED_OCR_MODEL_PATH, CONSERVATION_PATH, DNASE_DIR, LOOP_MODEL_PATH

from source.pretrained_model import anchor_predictor
from source.fine_tuned_ocr_model import ocr_predictor, ocr_models
from source.data_preprocess.loop_preprocessor import scan_CTCF_motif_on_anchor
from source.data_preprocess.motif_strength_adder import get_conservation, get_HS

def judge_motif_pattern(seq1, seq2):
    # CTCF_PWM & CTCF_PWN_REVERSE max similarity are both = 12.883604999999998
    score_threshold = 14
    sims_f = scan_CTCF_motif_on_anchor(seq1) # [[10.605276, 535, 'GGACCGGTAGGGGGCGCGC'],...]
    sims_r = scan_CTCF_motif_on_anchor(seq1, reverse=True)
    if len(sims_f) == 0 and len(sims_r) == 0:
        type1 = "x"
    elif len(sims_f) == 0:
        if sims_r[0][0] < score_threshold:
            type1 = "x"
        else:
            type1 = "r"
    elif len(sims_r) == 0:
        if sims_f[0][0] < score_threshold:
            type1 = "x"
        else:
            type1 = "f"
    else:
        if max(sims_f[0][0],sims_r[0][0]) < score_threshold:
            type1 = "x"
        elif sims_f[0][0] >= sims_r[0][0]:
            type1 = "f"
        else:
            type1 = "r"
    if type1 == "f":
        motif_strength_1 = sims_f[0][0]
    elif type1 == 'r':
        motif_strength_1 = sims_r[0][0]
    else:
        motif_strength_1 = 0

    sims_f = scan_CTCF_motif_on_anchor(seq2)
    # print(sims_f[:5])
    sims_r = scan_CTCF_motif_on_anchor(seq2, reverse=True)
    # print(sims_r[:5])
    if len(sims_f) == 0 and len(sims_r) == 0:
        type2 = "x"
    elif len(sims_f) == 0:
        if sims_r[0][0] < score_threshold:
            type2 = "x"
        else:
            type2 = "r"
    elif len(sims_r) == 0:
        if sims_f[0][0] < score_threshold:
            type2 = "x"
        else:
            type2 = "f"
    else:
        if max(sims_f[0][0],sims_r[0][0]) < score_threshold:
            type2 = "x"
        elif sims_f[0][0] >= sims_r[0][0]:
            type2 = "f"
        else:
            type2 = "r"
    if type2 == "f":
        motif_strength_2 = sims_f[0][0]
    elif type2 == 'r':
        motif_strength_2 = sims_r[0][0]
    else:
        motif_strength_2 = 0

    #     '''
    #     convergent, tandem, divergent, single, none是loop的两个anchors的CTCF-binding site在不同的motif方向（motif orientation）下的五种情况
    #     由于现在我只要两个anchors都有CTCF motif的环，所以目前有四类环，既可以convergent也可以tandem，convergent，tandem，divergent，others。
    #     设：convergent:1 tandem:2 divergent:3 single:4 none:5 两种类型都可能的就连续，如既可以convergent也可以tandem:12
    #     '''
    if type1 == "f" and type2 == "r":
        motif_pattern = 1
    elif (type1 == "f" and type2 == "f") or (type1 == "r" and type2 == "r"):
        motif_pattern = 2
    elif type1 == "r" and type2 == "f":
        motif_pattern = 3
    elif type1 == "x" and type2 == "x":
        motif_pattern = 5
    else:
        motif_pattern = 4
    return motif_pattern, motif_strength_1, motif_strength_2


def run_predictor(file_path):
    val_X = list()
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        fields = line.strip().split()
        chrom = fields[0]
        start1 = int(float(fields[1]))
        start2 = int(float(fields[2]))
        length = start2 - start1
        seq1 = fields[3]
        seq2 = fields[4]
        motif_pattern, motif_strength_1, motif_strength_2 = judge_motif_pattern(seq1, seq2)
        avg_motif_strength = (motif_strength_1 + motif_strength_2)/2.0
        std_motif_strength = np.std([motif_strength_1, motif_strength_2])
        con1 = get_conservation(cons_conn, chrom, (int(float(start1))-ext), (int(float(start1))+ext))
        con2 = get_conservation(cons_conn, chrom, (int(float(start2))-ext), (int(float(start2))+ext))
        avg_conservation = (con1+con2)/2.0
        std_conservation = np.std([con1, con2])
        HS_left = get_HS(dnase_conn, cell_line, chrom, (int(float(start1))-extension), (int(float(start1))+extension)) / (2*extension)
        HS_right = get_HS(dnase_conn, cell_line, chrom, (int(float(start2))-extension), (int(float(start2))+extension)) / (2*extension)
        avg_HS = (HS_left+HS_right)/2.0
        std_HS = np.std([HS_left, HS_right])
        HS_in_between = get_HS(dnase_conn, cell_line, chrom, (int(float(start1))), (int(float(start2)))) / (int(float(start2))-int(float(start1)))
        # DL score
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
        val_X.append([length, motif_pattern, avg_motif_strength, std_motif_strength, avg_conservation, std_conservation, avg_HS, std_HS, HS_in_between, HS_left, HS_right, avg_anchor_score, std_anchor_score, anchor_score1, anchor_score2, avg_ocr_score, std_ocr_score, ocr_score1, ocr_score2])


    best_model = joblib.load(LOOP_MODEL_PATH+'/model/RandomForestClassifier_mine_allfeats_8cellline_bychr.joblib')
    t = best_model.predict(val_X)
    print("result: ")
    print(t)
    predictions = best_model.predict_proba(val_X)
    print(predictions)
    # predictions = predictions[:, 1]

if __name__=="__main__":
    anchor_model, device, kmd = anchor_predictor.init_predictor(model_path=PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt')
    ocr_model, _, _ = ocr_predictor.init_predictor(
                            model_structure=ocr_models.OCRModel_v20230524_v1(),
                            model_path=FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_07_07_19_04_18.pt')

    cell_line = "GM12878"
    ext= 30 # for conservation
    extension = 2000 # for HS
    cons_conn = sqlite3.connect(CONSERVATION_PATH+'cons.db')
    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')

    # chrom = "chr1"
    # seq1 = 'CTAAAAATACAAAATTAGCCGGTCGTGGCGGCGCGCGCCTGTAATCCTAATTACTCGGGAGGCTGAGACAGGAGAATTACTTGAACCCAGGAGGCAGAGGTTGCAATGAGCCGAGATTGCACCACTGCACTCCAGCCTGGGCAACAAGCGCGAAACTCTGTCGCAAAAAATAAATAAATACATACATACATACATACAAATATTGCCAATTTGATAGGCAAAAATATATGCAAATTATGTTTTAACAAATATTAATAAAGTTGAACAATGGCCCAATGTAATTTGTTACCTATCTGAAAGACATGCACATTCAGGGAGTTACAGTACTTTTTCATGTCTTGCAGCATCAACAAATTGTTTTCTAAATGTGGAAAAAGAGACATTCAGTACAGGCTAACGTAACGTGGGACAGCATGCAAAGCCATGAGAAGTGGGAAAATGAGAGAGAATCTGGGCGTGGCTTACAGGGCTGCTGAAGTAAGACACACCCTTCGCTTCACCTGTGGCCTCCCGCGGCGGCTGCGGCTCCTTCGCAGGACCGGTAGGGGGCGCGCGCGGTTGAGTCCAGCAATTGCGAGGACCTCCGCAGGCGCAGCCCAGCACTGACGCCTCTCCGGGCCGTGGCTCCTCTTTCCCAGGCCGAAAGCGGCCGGGCCCTCTGCTCCGCTGCGCCCTGGCGCGGCCGCACCCACTGCGCGTTTGTGAGCTGGCCATCGGTCACACTGGGATACTCGAGGAGGAGGCCTGGGGCCTGCATTCAATTCCCTAAGAGGGACCCGCGCTGGCCGCTGCAGGCGGGTGGAGAACGGGCGTCTGAGTCTTTGGCTTTCGTACCGGAATCTCCCCGGGAGATTTTACAAGCCCCTCCGAATGTTCTGATTGAATTGGTCTGGGGGGGTCGGTCCGGGCATCTAACGAGCAGCCGGGATGGAGAATTACTAGTCTGGATTCTGGTCCTCAGAGGTTCACTCACGTTAATCAACCCGAGTCCATCTTGGTG'
    # kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
    # anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
    # seq2 = 'AGGCCACCAGGAGGCAGCA'

    file_path = sys.argv[1]
    run_predictor(file_path)

