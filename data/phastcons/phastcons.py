'''
@description: 将phastCons的数据转换为python可读的格式，染色体第一个碱基的索引是1，比如chr1就是[1,248956422]这个闭区间。
phastcons其实可以从这里下载http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.100way.phastCons/
但是我被误导了，从R下载的，然后转成python的格式，为了方便还放在了sqlite

straightly process chr*.phastCons100way.wigFix.gz program could find in CharID-add_feature.py or Lollipop-add_feature.py
'''

import numpy as np
import time
import sqlite3

CHROMOSOME_LIST = ['chr1',  'chr2',  'chr3',
                   'chr4',  'chr5',  'chr6',
                   'chr7',  'chr8',  'chr9',
                   'chr10', 'chr11', 'chr12',
                   'chr13', 'chr14', 'chr15',
                   'chr16', 'chr17', 'chr18',
                   'chr19', 'chr20', 'chr21',
                   'chr22', 'chrX',  'chrY']

def strip_lines(lines):
    new_lines = list()
    for line in lines:
        fields = line.strip().split(",")
        new_lines.append([fields[1].strip('"'), fields[2], fields[5].strip('"'), fields[6]])
    return new_lines

def strip_line(line):
    fields = line.strip().split(",")
    return [fields[1].strip('"'), fields[2], fields[5].strip('"'), fields[6]]

def process_a_file(f1_path, o1):
    f1 = open(f1_path, 'r')
    line = f1.readline()
    line = f1.readline()
    while line:
        o1.write("\t".join(strip_line(line))+"\n")
        line = f1.readline()
    f1.close()

def is_same(f1_path, f2_path):
    f1 = open(f1_path, 'r')
    f2 = open(f2_path, 'r')
    line1 = f1.readline()
    line2 = f2.readline()
    while line1 and line2:
        line1 = line1.strip().split()
        line1 = line1[0] + "\t" + line1[1] + "\t" + line1[3]
        line2 = line2.strip().split()
        line2 = line2[0] + "\t" + line2[1] + "\t" + line2[3]
        if line1 != line2:
            f1.close()
            f2.close()
            return False
        line1 = f1.readline()
        line2 = f2.readline()
    f1.close()
    f2.close()
    return True

def gen_3(name):
    o1 = open(f"./{name}.csv", 'w')
    process_a_file(f"./{name}1.csv", o1)
    process_a_file(f"./{name}2.csv", o1)
    process_a_file(f"./{name}3.csv", o1)
    o1.close()

def gen_2(name):
    o1 = open(f"./{name}.csv", 'w')
    process_a_file(f"./{name}1.csv", o1)
    process_a_file(f"./{name}2.csv", o1)
    o1.close()

def gen_1(name):
    o1 = open(f"./{name}.csv", 'w')
    process_a_file(f"./{name}1.csv", o1)
    o1.close()

def csv2numpy(name):
    '''
    trans NA to 0, and save as numpy
    '''
    cons_list = list()
    f1 = open(f"./{name}+.csv", 'r')
    line = f1.readline()
    while line:
        fields = line.strip().split()
        if fields[3] == "NA":
            cons_list.append(0.)
        else:
            cons_list.append(float(fields[3]))
        line = f1.readline()
    f1.close()
    cons_array = np.array(cons_list)
    np.save(f"./{name}.npy", cons_array)
    load_cons(name)

def load_cons(name):
    a = np.load(f"./{name}.npy")
    print(name, len(a))
    return a

def create_table():
    conn = sqlite3.connect('cons.db')
    print ("数据库打开成功")
    c = conn.cursor()
    for chr in CHROMOSOME_LIST:
        c.execute(f'''CREATE TABLE {chr}
            (POSITION    INT  PRIMARY KEY  NOT NULL,
            CONSERVATION    REAL    NOT NULL);''')
    print ("数据表创建成功")
    conn.commit()
    conn.close()

def insert():
    conn = sqlite3.connect('cons.db')
    c = conn.cursor()
    print ("数据库打开成功")

    for chr in CHROMOSOME_LIST:
        cons = load_cons(chr)
        for i in range(len(cons)):
            if cons[i] > 0:
                c.execute(f"INSERT INTO {chr} (POSITION,CONSERVATION) \
                    VALUES (?, ?)", (i, cons[i]))

        conn.commit()
    print ("数据插入成功")
    conn.close()

def search(start, end):
    conn = sqlite3.connect('cons.db')
    c = conn.cursor()
    print ("数据库打开成功")

    cursor = c.execute("SELECT sum(CONSERVATION) from chr1 where POSITION between (?) and (?);", (start, end))
    a = cursor.fetchall()
    print(a[0][0])
    print(type(a[0][0]))
    conn.close()

if __name__ == '__main__':
    start_time = time.time()

    # gen_3("chr1+")
    # gen_3("chr2+")
    # gen_2("chr3+")
    # gen_2("chr4+")
    # gen_2("chr5+")
    # gen_2("chr6+")
    # gen_2("chr7+")
    # gen_2("chr8+")
    # gen_2("chr9+")
    # gen_2("chr10+")
    # gen_2("chr11+")
    # gen_2("chr12+")
    # gen_2("chr13+")
    # gen_2("chr14+")
    # gen_2("chr15+")
    # gen_1("chr16+")
    # gen_1("chr17+")
    # gen_1("chr18+")
    # gen_1("chr19+")
    # gen_1("chr20+")
    # gen_1("chr21+")
    # gen_1("chr22+")
    # gen_2("chrX+")
    # gen_1("chrY+")

    # print(is_same("./chr2+.csv", "./chr2-.csv"))

    # csv2numpy("chr1")
    # csv2numpy("chr2")
    # csv2numpy("chr3")
    # csv2numpy("chr4")
    # csv2numpy("chr5")
    # csv2numpy("chr6")
    # csv2numpy("chr7")
    # csv2numpy("chr8")
    # csv2numpy("chr9")
    # csv2numpy("chr10")
    # csv2numpy("chr11")
    # csv2numpy("chr12")
    # csv2numpy("chr13")
    # csv2numpy("chr14")
    # csv2numpy("chr15")
    # csv2numpy("chr16")
    # csv2numpy("chr17")
    # csv2numpy("chr18")
    # csv2numpy("chr19")
    # csv2numpy("chr20")
    # csv2numpy("chr21")
    # csv2numpy("chr22")
    # csv2numpy("chrX")
    # csv2numpy("chrY")

    # create_table()
    # insert()
    search(990000, 1000000)

    print("Time taken: {}".format(time.time() - start_time))

