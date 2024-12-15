"""
@author: sjl
@purpose: preprocess DNase file download from ENCODE with 'bed narrowPeak' format.
"""

import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, re, sqlite3
from parameters import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils import TimeStatistic
from loop_preprocessor import load_UCSC_hg38
from motif_strength_adder import get_HS, get_HS_all

# @TimeStatistic
def make_DNase_statistics_for_one(file_path, draw=False):
    '''
    @purpose: get mean max min of open chromatin region len & signalValue in the narrowPeak bed
    @return: ocr_list, signalValue_list
    '''
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    ocr_list = list()
    signalValue_list = list()
    for i in lines:
        # a line data is like
        # chrom chromStart  chromEnd    name    score   strand  signalValue pValue  qValue  peak 
        # chr1  268011      268100      .       0       .       0.497607    -1      -1      75
        # the bed narrowPeak from ENCODE 
        fields = i.strip().split()
        templen = int(fields[2])-int(fields[1])
        ocr_list.append(templen)
        signalValue = float(fields[6])
        signalValue_list.append(signalValue)
        assert fields[3]==".", "name != ."
        assert int(fields[4])==0, "score != 0"
        assert fields[5]==".", "strand != ."
        assert int(fields[7])==-1, "pValue != -1"
        assert int(fields[8])==-1, "qValue != -1"
        assert int(fields[9])==75, "peak != 75"
    print(f"len: min={min(ocr_list)}, max={max(ocr_list)}, mean={np.mean(ocr_list)}, median={np.median(ocr_list)}")
    print(f"signalValue: min={min(signalValue_list)}, max={max(signalValue_list)}, mean={np.mean(signalValue_list)}, median={np.median(signalValue_list)}")
    if draw:
        FONTSIZE = 8
        plt.figure(dpi=300, figsize=(4, 3))
        plt.rc("font", family="Times New Roman")
        params = {"axes.titlesize": FONTSIZE,
                "legend.fontsize": FONTSIZE,
                "axes.labelsize": FONTSIZE,
                "xtick.labelsize": FONTSIZE,
                "ytick.labelsize": FONTSIZE,
                "figure.titlesize": FONTSIZE,
                "font.size": FONTSIZE}
        plt.rcParams.update(params)
        ax = plt.subplot(1, 2, 1)
        # plt.violinplot(ocr_list, showmeans=True, showmedians=True, showextrema=True)
        sns.violinplot(ocr_list)
        plt.xlabel('ocr len')
        # plt.ylabel('True Positive Rate (TPR)')
        plt.title('The statistics for DNase')
        # plt.legend(loc="best")
        ax = plt.subplot(1, 2, 2)
        # plt.violinplot(signalValue_list, showmeans=True, showmedians=True, showextrema=True)
        sns.violinplot(signalValue_list)
        plt.xlabel('signalValue len')
        plt.title('The statistics for DNase')
        # plt.show()
        plt.savefig(fname="dnase_statistics.png", format="png", bbox_inches="tight")
    return ocr_list, signalValue_list

@TimeStatistic
def make_DNase_statistics_for_many(dnase_accessions):
    '''
    @purpose: draw all cell line 的 ocr len & signalValue in a figure
    @parameters: dnase_accessions: a list of dnase_accessions [(cell_line, accession), ...]
    '''
    dnase_accessions = np.array(dnase_accessions)
    cell_line2order = {cell_line:i for i, cell_line in enumerate(dnase_accessions[:, 0])}
    # print(cell_line2order)
    cellLine2ocrLen_list = list()
    cellLine2signalValue_list = list()
    for cell_line, accession in dnase_accessions:
        print(f"Processing {cell_line}, {accession}")
        ocr_list, signalValue_list = make_DNase_statistics_for_one(file_path=ROOT_DIR+f"/data/DNase/{cell_line}/{accession}.bed")
        assert len(ocr_list) == len(signalValue_list)
        cellLine2ocrLen_list.extend([[cell_line2order[cell_line], i] for i in ocr_list])
        cellLine2signalValue_list.extend([[cell_line2order[cell_line], i] for i in signalValue_list])
    cellLine2ocrLen_list = pd.DataFrame(cellLine2ocrLen_list, columns=["cell_line", "ocr len"])
    cellLine2signalValue_list = pd.DataFrame(cellLine2signalValue_list, columns=["cell_line", "signalValue"])

    FONTSIZE = 8
    plt.figure(dpi=300, figsize=(16, 8))
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
    plt.rcParams.update(params)
    ax = plt.subplot(2, 1, 1)
    sns.violinplot(x="cell_line", y="ocr len", data=cellLine2ocrLen_list, scale="width")
    plt.axhline(100, ls="--", color="grey", lw=0.8)
    # the cell line from glob have already sorted, the sort here is just for secure
    cell_line_order_list = np.array(sorted([[i, cell_line2order[i]] for i in cell_line2order], key=lambda x: x[1]))
    plt.xticks(range(len(cell_line_order_list)), ["" for _ in range(len(cell_line_order_list))], fontsize=FONTSIZE)
    plt.ylim(0, 800)
    plt.xlabel('')
    plt.title('The statistics for DNase')
    ax = plt.subplot(2, 1, 2)
    sns.violinplot(x="cell_line", y="signalValue", data=cellLine2signalValue_list, scale="width")
    plt.axhline(0.35, ls="--", color="grey", lw=0.8)
    plt.xticks(range(len(cell_line_order_list)), cell_line_order_list[:, 0].tolist(), fontsize=FONTSIZE, rotation=315)
    plt.ylim(0, 10)
    # plt.xlabel('signalValue')
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.png", format="png", bbox_inches="tight")
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.eps", format="eps", bbox_inches="tight")

@TimeStatistic
def filter_narrowPeak_by_length_and_signalValue(dnase_accessions, ocr_length_threshold=100, signalValue_threshold=0.35):
    '''
    @purpose: use open chromatin region length & signalValue to filter high confidence peak,
              make peak region length >= ocr_length_threshold & peak signalValue >= signalValue_threshold in output file
    '''
    for cell_line, accession in dnase_accessions:
        f = open(DNASE_DIR+f"/{cell_line}/{accession}.bed", 'r')
        lines = f.readlines()
        f.close()
        f_out = open(DNASE_DIR+f"/{cell_line}/{accession}.filtered.bed", 'w')
        for i in lines:
            # chrom chromStart  chromEnd    name    score   strand  signalValue pValue  qValue  peak 
            fields = i.strip().split()
            temp_len = int(fields[2])-int(fields[1])
            temp_signalValue = float(fields[6])
            if temp_len < ocr_length_threshold or temp_signalValue < signalValue_threshold: continue
            f_out.write(i)
        f_out.close()

def create_table():
    conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    print ("数据库打开成功")
    c = conn.cursor()
    for cell_line in CELL_LINE_LIST:
        for chr in CHROMOSOME_LIST:
            c.execute(f'''DROP TABLE IF EXISTS {cell_line.replace('-', '_')}_{chr};''')
            c.execute(f'''CREATE TABLE {cell_line.replace('-', '_')}_{chr}
                (POSITION    INT  PRIMARY KEY  NOT NULL,
                SIGNALVALUE    REAL    NOT NULL);''')
    print ("数据表创建成功")
    conn.commit()
    conn.close()

def insert(dnase_accessions):
    conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    c = conn.cursor()
    print ("数据库打开成功")

    for cell_line, accession in dnase_accessions:
        f = open(DNASE_DIR+f"/{cell_line}/{accession}.bed", 'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            fields = l.strip().split()
            chr = fields[0]
            signalValue = float(fields[6])
            for pos in range(int(fields[1]), int(fields[2])):
                # if chr == "chr10":
                #     print(pos, signalValue)
                # print(f"INSERT INTO {cell_line.replace('-', '_')}_{chr} (POSITION,SIGNALVALUE) \
                #     VALUES ({pos}, {signalValue}) ON DUPLICATE KEY UPDATE SIGNALVALUE = {signalValue};")
                c.execute(f"INSERT OR IGNORE INTO {cell_line.replace('-', '_')}_{chr} (POSITION,SIGNALVALUE) \
                    VALUES ({pos}, {signalValue});")
        conn.commit()
        print (f"{cell_line}/{accession}.bed 数据插入成功")
    conn.close()

def search(cell_line, chr, start, end):
    conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    c = conn.cursor()
    print ("数据库打开成功")

    cursor = c.execute(f"SELECT sum(SIGNALVALUE) from {cell_line.replace('-', '_')}_{chr} where POSITION between {start} and {end};")
    a = cursor.fetchall()
    print(a[0][0])
    print(type(a[0][0]))
    conn.close()

def add_seq_to_bed(dnase_accessions, chromosome_dict):
    '''
    # output file fields
    # chrom chromStart chromEnd signalValue peak seq
    '''
    for cell_line, accession in dnase_accessions:
        print("add_seq_to_bed", cell_line, accession)
        f = open(DNASE_DIR+f"/{cell_line}/{accession}.bed", 'r')
        lines = f.readlines()
        f.close()
        f_out = open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR.bed", 'w')
        for i in lines:
            fields = i.strip().split()
            peak = int(fields[1])+int(fields[9])
            seq = chromosome_dict[fields[0]][peak-500:peak+500]
            f_out.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{fields[6]}\t{fields[9]}\t{seq}\n")
        f_out.close()

def gen_extension_ocr(cell_line_list):
    '''
    @purpose: generate extended open chromatin region with followed information:
              chrom chromStart chromEnd signalValue peak seq
              note that the signalValue is the average value of the extended region,
    '''
    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    extension = 500
    for cell_line in cell_line_list:
        print("gen_extension_ocr", cell_line)
        f = open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR.bed", 'r')
        lines = f.readlines()
        f.close()
        f_out = open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_ext.bed", 'w')
        ##  data from DNase bed file, the peak field is 75
        for i in lines:
            fields = i.strip().split()
            peak = int(fields[1])+int(fields[4])
            _HS = get_HS(dnase_conn, cell_line, fields[0], peak-extension, peak+extension) / (2*extension)
            f_out.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{_HS}\t{fields[4]}\t{fields[5]}\n")
        ## data from positive loop file, the peak field is 0
        for i in range(1, 6):
            f = open(POS_LOOP_PATH.format(cell_line, i), 'r')
            lines = f.readlines()
            f.close()
            f = open(POS_LOOP_PATH.format(cell_line, i), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                # anchor 1
                _HS = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension)) / (2*extension)
                f_out.write(f"{fields[0]}\t{int(float(fields[4]))-extension}\t{int(float(fields[4]))+extension}\t{_HS}\t0\t{fields[5]}\n")
                # anchor 2
                _HS = get_HS(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension)) / (2*extension)
                f_out.write(f"{fields[6]}\t{int(float(fields[10]))-extension}\t{int(float(fields[10]))+extension}\t{_HS}\t0\t{fields[11]}\n")
        ## data from negative loop file, the peak field is 0
        for typee in range(0, 6):
            for i in range(2):
                f = open(NEG_LOOP_PATH.format(cell_line, typee, i), 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    fields = line.strip().split()
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2:
                        continue
                    _HS = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension)) / (2*extension)
                    f_out.write(f"{fields[0]}\t{int(float(fields[4]))-extension}\t{int(float(fields[4]))+extension}\t{_HS}\t0\t{seq1}\n")
                    _HS = get_HS(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension)) / (2*extension)
                    f_out.write(f"{fields[6]}\t{int(float(fields[10]))-extension}\t{int(float(fields[10]))+extension}\t{_HS}\t0\t{seq2}\n")
        f_out.close()
    dnase_conn.close()

def gen_extension2_ocr(cell_line_list):
    '''
    @purpose: generate extended open chromatin region with followed information:
              chrom chromStart chromEnd signalValue_list peak seq
              note that the signalValue is the average value of the extended region,
              产生用于ocr训练的数据，
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
    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    extension = 500
    for cell_line in cell_line_list:
        print("gen_extension_ocr", cell_line)
        f = open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR.bed", 'r')
        lines = f.readlines()
        f.close()
        f_out = open(DNASE_DIR+f"/{cell_line}/{cell_line}_OCR_ext2.bed", 'w')
        ##  data from DNase bed file, the peak field is 75
        # for i in lines:
        #     fields = i.strip().split()
        #     peak = int(fields[1])+int(fields[4])
        #     _HS = get_HS_all(dnase_conn, cell_line, fields[0], peak-extension, peak+extension)
        #     _HS = ",".join([str(i) for i in _HS])
        #     f_out.write(f"{fields[0]}\t{fields[1]}\t{fields[2]}\t{_HS}\t{fields[4]}\t{fields[5]}\n")
        ## data from positive loop file, the peak field is 0
        for i in range(1, 6):
            f = open(POS_LOOP_PATH.format(cell_line, i), 'r')
            lines = f.readlines()
            f.close()
            f = open(POS_LOOP_PATH.format(cell_line, i), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                fields = line.strip().split()
                # 这里的1000bp是根据中点取的，注意pos_loop文件中的start和end是motif的start和end，所以此处要从中点拓展。
                # anchor 1
                _HS = get_HS_all(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension))
                _HS = ",".join([str(i) for i in _HS])
                f_out.write(f"{fields[0]}\t{int(float(fields[4]))-extension}\t{int(float(fields[4]))+extension}\t{_HS}\t0\t{fields[5]}\n")
                # anchor 2
                _HS = get_HS_all(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension))
                _HS = ",".join([str(i) for i in _HS])
                f_out.write(f"{fields[6]}\t{int(float(fields[10]))-extension}\t{int(float(fields[10]))+extension}\t{_HS}\t0\t{fields[11]}\n")
        ## data from negative loop file, the peak field is 0
        for typee in range(1, 6):
            for i in range(2):
                f = open(NEG_LOOP_PATH.format(cell_line, typee, i), 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    fields = line.strip().split()
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2:
                        continue
                    # anchor 1
                    _HS = get_HS_all(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension))
                    _HS = ",".join([str(i) for i in _HS])
                    f_out.write(f"{fields[0]}\t{int(float(fields[4]))-extension}\t{int(float(fields[4]))+extension}\t{_HS}\t0\t{seq1}\n")
                    # anchor 2
                    _HS = get_HS_all(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension))
                    _HS = ",".join([str(i) for i in _HS])
                    f_out.write(f"{fields[6]}\t{int(float(fields[10]))-extension}\t{int(float(fields[10]))+extension}\t{_HS}\t0\t{seq2}\n")
        f_out.close()
    dnase_conn.close()


if __name__=="__main__":
    seed = 9523
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性

    # make_DNase_statistics_for_one(file_path=ROOT_DIR+r"/data/DNase/GM12878/ENCFF759OLD.bed", draw=True)

    dnase_accessions = glob.glob(DNASE_DIR+r"/*/*.bed")
    dnase_accessions = [re.split(r"/|\.", repr(i).replace(r"\\", r'/'))[-3:-1] for i in dnase_accessions if 'filtered' not in i and "_" not in i] # split by / and .
    dnase_accessions = [i for i in dnase_accessions if i[0] in CELL_LINE_LIST] # i[0] is cell line, i[1] is accession
    # print(len(dnase_accessions))

    # make_DNase_statistics_for_many(dnase_accessions)

    # filter_narrowPeak_by_length_and_signalValue(dnase_accessions)

    # create_table()
    # insert(dnase_accessions)
    # search(cell_line="GM12878", chr="chr10", start=1000000, end=10000000)

    # hg38_chromosome_dict = load_UCSC_hg38()
    # add_seq_to_bed(dnase_accessions, hg38_chromosome_dict)

    # gen_extension_ocr(cell_line_list=["GM12878"])
    # gen_extension2_ocr(cell_line_list=["GM12878"])
    # gen_extension2_ocr(cell_line_list=["GM12878", "HCT116", "IMR-90", "K562"])
    gen_extension2_ocr(cell_line_list=["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"])
