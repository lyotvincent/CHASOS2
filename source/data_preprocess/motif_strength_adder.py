'''
@description: add
              ''
              avg_motif_strength 两个anchor上的motif的强度的平均值
              std_motif_strength 两个anchor上的motif的强度的标准差
              avg_conservation 两个anchor上的序列保守性的平均值
              std_conservation 两个anchor上的序列保守性的标准差
              avg_HS
              std_HS
              HS_in-between
              HS_left
              HS_right
              ''
              to positive loop
pos_loop_v2 columns:
0 chr1
1 start1
2 end1
3 orientation,strand
4 midpoint1
5 seq1
6 chr2
7 start2
8 end2
9 orientation,strand
10 midpoint2
11 seq2
12 avg_motif_strength 两个anchor上的motif的强度的平均值
13 std_motif_strength 两个anchor上的motif的强度的标准差
14 avg_conservation 两个anchor上的序列保守性的平均值
15 std_conservation 两个anchor上的序列保守性的标准差
16 avg_HS
17 std_HS
18 HS_in-between
19 HS_left
20 HS_right
'''


import sys, os, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from parameters import *
import numpy as np
import copy


def get_motif_strength():
    motif_strength_dict = dict()

    f = open(FIMO_CUSTOMISED_RESULT_PATH.format(""), 'r')
    line = f.readline()
    line = f.readline()
    while line:
        fields = line.strip().split()
        motif_strength_dict[fields[2]+"_"+fields[5]+"_"+str(int(fields[3])-1)+"_"+str(int(fields[4])-1)] = fields[6]
        line = f.readline()
    f.close()
    f = open(FIMO_CUSTOMISED_RESULT_PATH.format("_reverse"), 'r')
    line = f.readline()
    line = f.readline()
    while line:
        fields = line.strip().split()
        motif_strength_dict[fields[2]+"_"+fields[5]+"_"+str(int(fields[3])-1)+"_"+str(int(fields[4])-1)] = fields[6]
        line = f.readline()
    f.close()
    
    '''
    return: a dict of motif strength
    key: chr_strand_start_end (e.g. chr1_+_100_200)
         note that the start and end are 0-based, 即从0开始算起
    value: motif strength, the score in FIMO result
    '''
    return motif_strength_dict

def get_conservation(conn, chr, start, end):
    '''
    return the feature of sequence conservation on anchors.
    the db is created in /data/phastcons/phastcons.py
    '''
    # conn = sqlite3.connect(CONSERVATION_PATH+'cons.db')
    c = conn.cursor()
    # print ("数据库打开成功")
    cursor = c.execute(f"SELECT sum(CONSERVATION) from {chr} where POSITION between (?) and (?);", (start, end))
    a = cursor.fetchall()
    # print(a[0][0])
    # print(type(a[0][0]))
    # conn.close()
    if a[0][0] == None:
        return 0.
    else:
        return a[0][0]

def get_HS(conn, cell_line, chr, start, end):
    '''
    return the feature of DNase hypersensitive sites on anchors.
    the db is created in /source/data_preprocess/dnase_preprocessor.py
    '''
    # conn = sqlite3.connect('cons.db')
    c = conn.cursor()
    # print ("数据库打开成功")

    cursor = c.execute(f"SELECT sum(SIGNALVALUE) from {cell_line.replace('-', '_')}_{chr} where POSITION between {start} and {end};")
    a = cursor.fetchall()
    # print(a[0][0])
    # print(type(a[0][0]))
    # conn.close()
    if a[0][0] == None:
        return 0.
    else:
        return a[0][0]

def get_HS_all(conn, cell_line, chr, start, end, length=1000):
    '''
    return the feature of DNase hypersensitive sites on anchors.
    the db is created in /source/data_preprocess/dnase_preprocessor.py
    '''
    signalValue_list = [0.]*length
    # conn = sqlite3.connect('cons.db')
    c = conn.cursor()
    # print ("数据库打开成功")

    cursor = c.execute(f"SELECT POSITION, SIGNALVALUE from {cell_line.replace('-', '_')}_{chr} where POSITION between {start} and {end} ORDER BY POSITION;")
    a = cursor.fetchall()
    for record in a:
        if int(record[0])-start >= length:
            break
        signalValue_list[int(record[0])-start] = float(record[1])
    # print(a[0][0])
    # print(type(a[0][0]))
    # conn.close()
    return signalValue_list

def add_new_features_to_pos_loop(cell_line_list, motif_strength_dict):
    cons_conn = sqlite3.connect(CONSERVATION_PATH+'cons.db')
    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    ext=30 # for conservation
    extension = 2000 # for HS
    for cell_line in cell_line_list:
        for i in range(1, 6):
            f = open(POS_LOOP_PATH.format(cell_line, i), 'r')
            lines = f.readlines()
            f.close()
            f = open(POS_LOOP_PATH.format(cell_line, i)+'_v2', 'w')
            for line in lines:
                fields = line.strip().split()
                # anchor 1
                if fields[3] == '.,.':
                    motif_strength_1 = 0.
                else:
                    motif_strength_1 = float(motif_strength_dict[fields[0]+"_"+fields[3].split(",")[1]+"_"+fields[1]+"_"+fields[2]])
                con1 = get_conservation(cons_conn, fields[0], (int(float(fields[4]))-ext), (int(float(fields[4]))+ext))
                HS_left = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension)) / (2*extension)
                # anchor 2
                if fields[9] == '.,.':
                    motif_strength_2 = 0.
                else:
                    motif_strength_2 = float(motif_strength_dict[fields[6]+"_"+fields[9].split(",")[1]+"_"+fields[7]+"_"+fields[8]])
                con2 = get_conservation(cons_conn, fields[6], (int(float(fields[10]))-ext), (int(float(fields[10]))+ext))
                HS_right = get_HS(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension)) / (2*extension)
                # compute new features
                avg_motif_strength = (motif_strength_1 + motif_strength_2)/2.0
                std_motif_strength = np.std([motif_strength_1, motif_strength_2])
                avg_conservation = (con1+con2)/2.0
                std_conservation = np.std([con1, con2])
                avg_HS = (HS_left+HS_right)/2.0
                std_HS = np.std([HS_left, HS_right])
                HS_in_between = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))), (int(float(fields[10])))) / (int(float(fields[10]))-int(float(fields[4])))
                # new fields 这些都是新加的特征
                new_fields = copy.deepcopy(fields)
                new_fields.insert(12, str(avg_motif_strength))
                new_fields.insert(13, str(std_motif_strength))
                new_fields.insert(14, str(avg_conservation))
                new_fields.insert(15, str(std_conservation))
                new_fields.insert(16, str(avg_HS))
                new_fields.insert(17, str(std_HS))
                new_fields.insert(18, str(HS_in_between))
                new_fields.insert(19, str(HS_left))
                new_fields.insert(20, str(HS_right))
                f.write("\t".join(new_fields)+"\n")
            f.close()
    cons_conn.close()
    dnase_conn.close()

def add_new_features_to_neg_loop(cell_line_list, motif_strength_dict):
    cons_conn = sqlite3.connect(CONSERVATION_PATH+'cons.db')
    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    ext=30 # for conservation
    extension = 2000 # for HS
    for cell_line in cell_line_list:
        for typee in range(0, 6):
            for i in range(2):
                f = open(NEG_LOOP_PATH.format(cell_line, typee, i), 'r')
                lines = f.readlines()
                f.close()
                f = open(NEG_LOOP_PATH.format(cell_line, typee, i)+'_v2', 'w')
                for line in lines:
                    fields = line.strip().split()
                    seq1 = fields[5]
                    seq2 = fields[11]
                    if "N" in seq1 or "N" in seq2:
                        continue
                    # anchor 1
                    if fields[3] == '.,.':
                        motif_strength_1 = 0.
                    else:
                        motif_strength_1 = float(motif_strength_dict[fields[0]+"_"+fields[3].split(",")[1]+"_"+fields[1]+"_"+fields[2]])
                    con1 = get_conservation(cons_conn, fields[0], (int(float(fields[4]))-ext), (int(float(fields[4]))+ext))
                    HS_left = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))-extension), (int(float(fields[4]))+extension)) / (2*extension)
                    # anchor 2
                    if fields[9] == '.,.':
                        motif_strength_2 = 0.
                    else:
                        motif_strength_2 = float(motif_strength_dict[fields[6]+"_"+fields[9].split(",")[1]+"_"+fields[7]+"_"+fields[8]])
                    con2 = get_conservation(cons_conn, fields[6], (int(float(fields[10]))-ext), (int(float(fields[10]))+ext))
                    HS_right = get_HS(dnase_conn, cell_line, fields[6], (int(float(fields[10]))-extension), (int(float(fields[10]))+extension)) / (2*extension)
                    # compute new features
                    avg_motif_strength = (motif_strength_1 + motif_strength_2)/2.0
                    std_motif_strength = np.std([motif_strength_1, motif_strength_2])
                    avg_conservation = (con1+con2)/2.0
                    std_conservation = np.std([con1, con2])
                    avg_HS = (HS_left+HS_right)/2.0
                    std_HS = np.std([HS_left, HS_right])
                    HS_in_between = get_HS(dnase_conn, cell_line, fields[0], (int(float(fields[4]))), (int(float(fields[10])))) / (int(float(fields[10]))-int(float(fields[4])))
                    # new fields
                    new_fields = copy.deepcopy(fields)
                    new_fields.insert(12, str(avg_motif_strength))
                    new_fields.insert(13, str(std_motif_strength))
                    new_fields.insert(14, str(avg_conservation))
                    new_fields.insert(15, str(std_conservation))
                    new_fields.insert(16, str(avg_HS))
                    new_fields.insert(17, str(std_HS))
                    new_fields.insert(18, str(HS_in_between))
                    new_fields.insert(19, str(HS_left))
                    new_fields.insert(20, str(HS_right))
                    f.write("\t".join(new_fields)+"\n")
                f.close()
    cons_conn.close()
    dnase_conn.close()
if __name__ == "__main__":
    time1 = time.time()
    # print(FIMO_CUSTOMISED_RESULT_PATH)
    # motif_strength_dict = get_motif_strength()
    # print(motif_strength_dict["chr1_+_11398_11431"])

    # conservation = get_conservation('chr1', 990000, 1000000)
    # print(conservation)

    # add_new_features_to_pos_loop(["GM12878"], motif_strength_dict)
    # add_new_features_to_pos_loop(CELL_LINE_LIST, motif_strength_dict)
    # add_new_features_to_neg_loop(["GM12878"], motif_strength_dict)
    # add_new_features_to_neg_loop(CELL_LINE_LIST, motif_strength_dict)

    print("time cost: ", time.time()-time1, "s")

    dnase_conn = sqlite3.connect(DNASE_DIR+'/dnase.db')
    a = get_HS_all(dnase_conn, "GM12878", "chr1", 268000, 269000)
    print(a)

