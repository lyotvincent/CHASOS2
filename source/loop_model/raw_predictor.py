
import sys, os, json, math, sqlite3, joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocess.motif_strength_adder import get_motif_strength, get_conservation, get_HS
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import *
from data_preprocess.loop_preprocessor import load_UCSC_hg38
from pretrained_model import anchor_predictor
from fine_tuned_ocr_model import ocr_predictor, ocr_models

cons_conn = sqlite3.connect(CONSERVATION_PATH+'cons.db')
dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
anchor_model, device, kmd = anchor_predictor.init_predictor(model_path=PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt')
ocr_model, _, _ = ocr_predictor.init_predictor(
                        model_structure=ocr_models.OCRModel_v20230524_v1(),
                        model_path=FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_07_07_19_04_18.pt'
                    )
CHROMOSOME_DICT = load_UCSC_hg38()

cell_line = "K562"
chr = "chr1"
anchor_pos_list = [
["peak_374", "peak_378",18597564,18597597,"+",18820941,18820974,"-",4],
["peak_374", "peak_380",18597564,18597597,"+",18875395,18875428,"-",4],
["peak_374", "peak_383",18597564,18597597,"+",18913309,18913342,"+",4],
["peak_374", "peak_401",18597564,18597597,"+",19661983,19662016,".",1],
["peak_378", "peak_380",18820941,18820974,"-",18875395,18875428,"-",3],
["peak_380", "peak_382",18875395,18875428,"-",18907035,18907068,"-",3],
["peak_381", "peak_382",18884290,18884323,"+",18907035,18907068,"-",4],
["peak_381", "peak_383",18884290,18884323,"+",18913309,18913342,"+",4],
["peak_381", "peak_389",18884290,18884323,"+",19241415,19241448,".",1],
["peak_382", "peak_383",18907301,18907334,"+",18913309,18913342,"+",4],
["peak_382", "peak_389",18907301,18907334,"+",19241415,19241448,".",1],
["peak_382", "peak_399",18907301,18907334,"+",19653861,19653894,"-",4],
["peak_382", "peak_401",18907301,18907334,"+",19661983,19662016,".",1],
["peak_383", "peak_398",18913244,18913277,"+",19651302,19651335,".",1],
["peak_392", "peak_398",19391406,19391439,".",19651302,19651335,".",0],
["peak_392", "peak_399",19391406,19391439,".",19653861,19653894,"-",1],
["peak_398", "peak_399",19651302,19651335,".",19653861,19653894,"-",1],
["peak_399", "peak_401",19653861,19653894,"-",19661983,19662016,".",1],]
# ["peak_408", "peak_411",19918161,19918194,"+",20152567,20152600,".",1],
# ["peak_408", "peak_421",19918161,19918194,"+",20493963,20493996,"-",4],
# ["peak_408", "peak_424",19918161,19918194,"+",20633415,20633448,"+",4],
# ["peak_409", "peak_411",19974943,19974976,"+",20152567,20152600,".",1],
# ["peak_409", "peak_421",19974943,19974976,"+",20493963,20493996,"-",4]]
# 有.的都是没有CTCF的，他们的motif位置都是取的anchor中心
# peak_374	peak_378    18597564,18597597,f,+	18820941,18820974,f,-    4
# peak_374	peak_380    18597564,18597597,f,+	18875395,18875428,f,-    4
# peak_374	peak_383    18597564,18597597,f,+	18913309,18913342,r,+    4
# peak_374	peak_401    18597564,18597597,f,+	19661983,19662016,.      1
# peak_378	peak_380    18820941,18820974,f,-	18875395,18875428,f,-    3
# peak_380	peak_382    18875395,18875428,f,-	18907035,18907068,f,-    3
# peak_381	peak_382    18884290,18884323,f,+	18907035,18907068,f,-    4
# peak_381	peak_383    18884290,18884323,f,+	18913309,18913342,r,+    4
# peak_381	peak_389    18884290,18884323,f,+	19241415,19241448,.      1
# peak_382	peak_383    18907301,18907334,f,+	18913309,18913342,r,+    4
# peak_382	peak_389    18907301,18907334,f,+	19241415,19241448,.      1
# peak_382	peak_399    18907301,18907334,f,+	19653861,19653894,f,-    4
# peak_382	peak_401    18907301,18907334,f,+	19661983,19662016,.      1
# peak_383	peak_398    18913244,18913277,f,+	19651302,19651335,.      1
# peak_392	peak_398    19391406,19391439,.	    19651302,19651335,.      0
# peak_392	peak_399    19391406,19391439,.	    19653861,19653894,f,-    1
# peak_398	peak_399    19651302,19651335,.     19653861,19653894,f,-    1
# peak_399	peak_401    19653861,19653894,f,-	19661983,19662016,.      1
# peak_408	peak_411    19918161,19918194,f,+	20152567,20152600,.      1
# peak_408	peak_421    19918161,19918194,f,+	20493963,20493996,f,-    4
# peak_408	peak_424	19918161,19918194,f,+	20633415,20633448,r,+    4
# peak_409	peak_411    19974943,19974976,f,+	20152567,20152600,.      1
# peak_409	peak_421    19974943,19974976,f,+	20493963,20493996,f,-    4

# peak_374 > 1.854278728606357  17.0248
# peak_378 < 56.491075794621025 24.2893
# peak_380 < 69.78117359413203  24.6694
# peak_381 > 71.96026894865525  19.8099
# peak_382 < 77.57322738386308  17.6446
# peak_383 < 79.04511002444988  18.5207
# peak_389
# peak_398
# peak_399 < 260.10378973105134 19.4132
# peak_401
# peak_408 > 324.7455990220049  17.2066
# peak_409 > 338.6312958435208  18.2479
# peak_411
# peak_421 < 465.5223716381418  31.2397
# peak_424 < 499.65953545232276 15.0496

# anchor1_CTCF_start, anchor1_CTCF_end, anchor2_CTCF_start, anchor2_CTCF_end = anchor_pos_list[0]

def predict(anchor1_CTCF_start, anchor1_CTCF_end, strand1, anchor2_CTCF_start, anchor2_CTCF_end, strand2, motif_pattern):
    # same to motif_strength_adder.py
    ext=30 # for conservation
    extension = 2000 # for HS

    motif_strength_dict = get_motif_strength()
    if strand1 == ".":
        motif_strength_1 = 0.
    else:
        motif_strength_1 = float(motif_strength_dict[f"{chr}_{strand1}_{anchor1_CTCF_start}_{anchor1_CTCF_end}"])
    if strand2 == ".":
        motif_strength_2 = 0.
    else:
        motif_strength_2 = float(motif_strength_dict[f"{chr}_{strand2}_{anchor2_CTCF_start}_{anchor2_CTCF_end}"])

    conservation1 = get_conservation(cons_conn,  chr=chr, start=anchor1_CTCF_start-13, end=anchor1_CTCF_end+13)
    conservation2 = get_conservation(cons_conn,  chr=chr, start=anchor2_CTCF_start-13, end=anchor2_CTCF_end+13)

    HS_left = get_HS(dnase_conn, cell_line=cell_line, chr=chr, start=anchor1_CTCF_start-983, end=anchor1_CTCF_end+983) / extension
    HS_right = get_HS(dnase_conn, cell_line=cell_line, chr=chr, start=anchor2_CTCF_start-983, end=anchor2_CTCF_end+983) / extension

    seq1 = CHROMOSOME_DICT[chr][anchor1_CTCF_start-483: anchor1_CTCF_end+484]
    seq2 = CHROMOSOME_DICT[chr][anchor2_CTCF_start-483: anchor2_CTCF_end+484]

    length = anchor2_CTCF_end - anchor1_CTCF_start - 34
    motif_pattern = motif_pattern # convergent:4 tandem:3 divergent:2 single:1 none:0
    avg_motif_strength = (motif_strength_1 + motif_strength_2)/2.
    std_motif_strength = np.std([motif_strength_1, motif_strength_2])
    avg_conservation = (float(conservation1) + float(conservation2))/2.
    std_conservation = np.std([conservation1, conservation2])
    avg_HS = (float(HS_left) + float(HS_right))/2.
    std_HS = np.std([HS_left, HS_right])
    HS_in_between = get_HS(dnase_conn, cell_line=cell_line, chr=chr, start=anchor1_CTCF_start+13, end=anchor2_CTCF_end-13) / length

    kmers1 = anchor_predictor.encode_kmer(kmd, [seq1])
    kmers2 = ocr_predictor.encode_kmer(kmd, [seq2])
    anchor_score1 = anchor_predictor.get_pred_scores(anchor_model, kmers1, device)[0][0].item()
    anchor_score2 = anchor_predictor.get_pred_scores(anchor_model, kmers2, device)[0][0].item()
    avg_anchor_score = (anchor_score1 + anchor_score2)/2.
    std_anchor_score = np.std([anchor_score1, anchor_score2])
    ocr_score1 = ocr_predictor.get_pred_scores(ocr_model, kmers1, device)[0].mean().item()
    ocr_score2 = ocr_predictor.get_pred_scores(ocr_model, kmers2, device)[0].mean().item()
    avg_ocr_score = (ocr_score1 + ocr_score2)/2.
    std_ocr_score = np.std([ocr_score1, ocr_score2])


    val_X = np.array([length, motif_pattern, avg_motif_strength, std_motif_strength, avg_conservation, std_conservation, avg_HS, std_HS, HS_in_between, HS_left, HS_right, avg_anchor_score, std_anchor_score, anchor_score1, anchor_score2, avg_ocr_score, std_ocr_score, ocr_score1, ocr_score2]).reshape(1, -1)
    best_model = joblib.load(LOOP_MODEL_PATH+'/model/RandomForestClassifier_mine_allfeats_8cellline_byhealth.joblib')
    predictions = best_model.predict_proba(val_X)
    print(predictions[:, 1])

    val_X = np.array([length, motif_pattern, avg_motif_strength, std_motif_strength, avg_conservation, std_conservation, avg_HS, std_HS, HS_in_between, HS_left, HS_right]).reshape(1, -1)
    basefunctional_model = joblib.load(LOOP_MODEL_PATH+'/model/RandomForestClassifier_mine_basefunctional_8cellline_byhealth.joblib')
    predictions = basefunctional_model.predict_proba(val_X)
    print(predictions[:, 1])

def compute_pos():
    f = open(CHIA_PET2_RESULT_PATH.format("K562")+"/K562.significant.interactions.intra.filtered.MICC", "r")
    lines = f.readlines()
    f.close()
    lines = lines[157:180]
    genome_start = 18590000
    genome_end = 20635000
    # print(lines[-1])
    for line in lines:
        fields = line.strip().split("\t")
        anchor1_start = int(fields[1])
        anchor1_end = int(fields[2])
        anchor2_start = int(fields[4])
        anchor2_end = int(fields[5])
        anchor1 = (anchor1_start+anchor1_end)/2.
        anchor2 = (anchor2_start+anchor2_end)/2.
        anchor1_pos = 500*(anchor1 - genome_start) / (genome_end - genome_start)
        anchor2_pos = 500*(anchor2 - genome_start) / (genome_end - genome_start)
        print(fields[6], fields[7], anchor1_pos, anchor2_pos, anchor2_pos-anchor1_pos)

def compute_motif_strength(anchor1_CTCF_start, anchor1_CTCF_end, strand1):
    
    motif_strength_dict = get_motif_strength()
    if strand1 == ".":
        motif_strength_1 = 0.
    else:
        motif_strength_1 = float(motif_strength_dict[f"{chr}_{strand1}_{anchor1_CTCF_start}_{anchor1_CTCF_end}"])
    print(motif_strength_1)
    return motif_strength_1

'''
motif_strength是FIMO结果里的score
'''
def compute_motif_strength_2(start=18590000, end=19670000):
    f = open(FIMO_CUSTOMISED_RESULT_PATH.format(""), 'r')
    lines = f.readlines()
    f.close()
    f = open(FIMO_CUSTOMISED_RESULT_PATH.format("_reverse"), 'r')
    lines2 = f.readlines()
    f.close()
    lines = lines[1:]+lines2[1:]
    new_lines = list()
    score_list = list()
    pos_list = list()
    for l in lines:
        fields = l.strip().split()
        if fields[2]=="chr1" and int(fields[3])>=start and int(fields[4])<=end:
            new_lines.append([fields[3], fields[4], fields[6]])
            pos_list.append(int(fields[3]))
            score_list.append(float(fields[6]))
            # print(l)
    import matplotlib.pyplot as plt
    print(min(score_list), max(score_list))
    plt.figure(dpi=300, figsize=(10, 2))
    plt.bar(pos_list, score_list, width=1000, color="#ED1C24")
    plt.xlim(18590000, 19670000)
    plt.ylim(0, 32)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    # plt.savefig('output.png', dpi=300)


def compute_conservation(start=18590000, end=19670000):
    import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize']=(12.8, 7.2)
    conservation_list = list()
    pos_list = list()
    for i in range(start, end, 1000):
        pos_list.append(i)
        conservation = get_conservation(cons_conn,  chr=chr, start=i, end=i+1000)/1000.
        conservation_list.append(float(conservation))
    print(min(conservation_list), max(conservation_list))
    plt.figure(dpi=300, figsize=(10, 2))
    plt.bar(pos_list, conservation_list, width=1000, color="#ED1C24")
    # plt.title('t-SNE visualization of 6645 data points')
    plt.xlim(18590000, 19670000)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    # plt.savefig('output.png', dpi=300)

def compute_HS(start=18590000, end=19670000):
    import matplotlib.pyplot as plt
    hs_list = list()
    pos_list = list()
    for i in range(start, end, 1000):
        pos_list.append(i)
        hs = get_HS(dnase_conn, cell_line=cell_line, chr=chr, start=i, end=i+1000)/1000
        hs_list.append(float(hs))
    print(min(hs_list), max(hs_list))
    plt.figure(dpi=300, figsize=(10, 2))
    plt.bar(pos_list, hs_list, width=1000, color="#ED1C24")
    plt.xlim(18590000, 19670000)
    # plt.ylim(0, 1)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def compute_anchor_score(start=18590000, end=19670000):
    import matplotlib.pyplot as plt
    anchor_score_list = list()
    pos_list = list()
    for i in range(start, end, 1000):
        pos_list.append(i)
        seq = CHROMOSOME_DICT[chr][i: i+1000]
        kmers = anchor_predictor.encode_kmer(kmd, [seq])
        anchor_score = anchor_predictor.get_pred_scores(anchor_model, kmers, device)[0][0].item()
        anchor_score_list.append(float(anchor_score))
    print(min(anchor_score_list), max(anchor_score_list))
    plt.figure(dpi=300, figsize=(10, 2))
    plt.bar(pos_list, anchor_score_list, width=1000, color="#00A2E8")
    plt.xlim(18590000, 19670000)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def compute_ocr_score(start=18590000, end=19670000):
    import matplotlib.pyplot as plt
    ocr_score_list = list()
    pos_list = list()
    for i in range(start, end, 1000):
        pos_list.append(i)
        seq = CHROMOSOME_DICT[chr][i: i+1000]
        kmers = ocr_predictor.encode_kmer(kmd, [seq])
        ocr_score = ocr_predictor.get_pred_scores(ocr_model, kmers, device)[0].mean().item()
        ocr_score_list.append(float(ocr_score))
    print(min(ocr_score_list), max(ocr_score_list))
    plt.figure(dpi=300, figsize=(10, 2))
    plt.bar(pos_list, ocr_score_list, width=1000, color="#00A2E8")
    plt.xlim(18590000, 19670000)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if "__main__" == __name__:
    # predict()
    # compute_pos()

    # predict loop probability
    # for peak1, peak2, anchor1_CTCF_start, anchor1_CTCF_end, strand1, anchor2_CTCF_start, anchor2_CTCF_end, strand2, motif_pattern in anchor_pos_list:
    #     print(peak1, peak2)
    #     predict(anchor1_CTCF_start, anchor1_CTCF_end, strand1, anchor2_CTCF_start, anchor2_CTCF_end, strand2, motif_pattern)
    #     print("------------------------------------------------------")

    # compute_motif_strength
    # motif_list = set() # [peak_number, start, end, strand]
    # for peak1, peak2, anchor1_CTCF_start, anchor1_CTCF_end, strand1, anchor2_CTCF_start, anchor2_CTCF_end, strand2, motif_pattern in anchor_pos_list:
    #     # print(peak1, peak2)
    #     motif_list.add((peak1, anchor1_CTCF_start, anchor1_CTCF_end, strand1))
    #     motif_list.add((peak2, anchor2_CTCF_start, anchor2_CTCF_end, strand2))
    # motif_list = list(motif_list)
    # motif_list.sort()
    # for peak, start, end, strand in motif_list:
    #     print(peak, start, end, strand)
    #     compute_motif_strength(start, end, strand)
    #     print("------------------------------------------------------")

    compute_motif_strength_2()
    compute_conservation()
    compute_HS()
    compute_anchor_score()
    compute_ocr_score()


