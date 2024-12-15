#encoding:utf-8
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import numpy as np
import os
import sys
import math
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from source.parameters import FIMO_CUSTOMISED_RESULT_PATH, NEG_LOOP_PATH, POS_LOOP_PATH
from source.data_preprocess.loop_preprocessor import load_UCSC_hg38
from source.loop_model.preprocess_data import judge_loop_type

import processSeq
import warnings
from sklearn import preprocessing
import sklearn.preprocessing
from gensim import corpora, models, similarities

def getkmervec(DNAdata1,DNAdata2,kmerdict):
    counter = 0
    DNAFeatureVecs = np.zeros((len(DNAdata1),2*len(kmerdict.keys())), dtype="float32")
    
    for DNA in DNAdata1:
        if counter % 1000 == 0:
            print("DNA %d of %d\r" % (counter, len(DNAdata1)))
            sys.stdout.flush()

        for word in DNA:
            DNAFeatureVecs[counter][kmerdict[word]] += 1
        counter += 1
    
    counter = 0
    for DNA in DNAdata2:
        if counter % 1000 == 0:
            print("DNA %d of %d\r" % (counter, len(DNAdata2)))
            sys.stdout.flush()
        for word in DNA:
            DNAFeatureVecs[counter][kmerdict[word]+len(kmerdict.keys())] += 1
        counter += 1

    return DNAFeatureVecs

def genkmerdict(k):
    vocab = ['A','C','T','G']
    dict1 = {}
    result = [""]
    for i in range(k):
        temp_result = []
        for word in result:
            if len(word) != i:
                print(word)
                print(len(word),i)
                continue
            else:
                for letter in vocab:
                    temp_word = word + letter
                    temp_result.append(temp_word)

        result = temp_result

    count = 0
    for word in result:
        dict1[word] = count 
        count += 1

    return dict1

def Unsupervised(Range,word):
    unlabelfile = open("./Temp/Unsupervised","w")
    # CTCF = pd.read_table("../Data/%s/fimo2.csv" %(cell),sep = ",")
    CTCF = pd.read_table(FIMO_CUSTOMISED_RESULT_PATH.format(""))

    list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
            "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]

    # dict_pos={}
    dict_pos=load_UCSC_hg38()
    total = len(CTCF)
    print(total)
    for i in range(total):
        if (i % 100 == 0) and (i != 0):
            print("number of unsupervised: %d of %d\r" %(i,total))
            sys.stdout.flush()

        start = CTCF["start"][i] - 1 - Range
        end = CTCF["stop"][i] + Range
        chromosome = CTCF["sequence_name"][i]
        if chromosome not in list:
            continue
        # if chromosome in dict_pos.keys():
        #     strs = dict_pos[chromosome]
        # else:
        #     strs = processSeq.getString("../../Chromosome/" + chromosome + ".fasta")
        #     dict_pos[chromosome] = strs
        strs = dict_pos[chromosome]
        strand = CTCF["strand"][i]
        edstrs = strs[start:end]
        print(end-start)

        if strand == "-":
            edstrs = edstrs[::-1]
            edstrs = processSeq.get_reverse_str(edstrs)

        if "N" in edstrs:
            continue

        sentence = processSeq.DNA2Sentence(edstrs,word)
        unlabelfile.write(str(sentence)+"\n")
    unlabelfile.close()

def gen_Seq(Range,cell,direction):
    print("Generating Seq...")
    table = pd.read_table("./Temp/%s/%s/LabelData.csv" %(cell,direction), sep=',')
    print(len(table))
    table.drop_duplicates()
    print(len(table))
    label_file = open("./Temp/%s/%s/LabelSeq" %(cell,direction), "w")

    label_file.write("label\t"
                  "seq1\t"
                  "strand1\t"
                  "seq2\t"
                  "strand2\t"
                  "chr\t"
                  "motif1\t"
                  "motif2\n")

    total = len(table)

    list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
            "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]
    number_positive = 0
    dict_pos={}

    cg1 = np.zeros((total),dtype="float32")
    cg2 = np.zeros((total),dtype="float32")
    cgm = np.zeros((total),dtype="float32")
    chr_length = np.zeros((total),dtype = "int64")
    for i in range(total):
        
        if (number_positive % 100 == 0) and (number_positive != 0):
            print("number of seq: %d of %d\r" %(number_positive,total))
            sys.stdout.flush()

        chromosome = table["chromosome"][i]
        if chromosome in dict_pos.keys():
            strs = dict_pos[chromosome]
        else:
            strs = processSeq.getString("../../Chromosome/" + str(chromosome) + ".fasta")
            dict_pos[chromosome] = strs

        start1 = int(table["C1_start"][i] - 1 - Range)
        end1 = int(table["C1_end"][i] + Range)
        start2 = int(table["C2_start"][i] - 1 - Range)
        end2 = int(table["C2_end"][i] + Range)

        motif1 = table["C1_motif"][i]
        motif2 = table["C2_motif"][i]
        strand1 = table["C1_strand"][i]
        strand2 = table["C2_strand"][i]
        
        edstrs1 = strs[start1 : end1]
        edstrs2 = strs[start2 : end2]
        edstrs3 = strs[end1 : start2]

        
        chr_length[number_positive] = len(strs)

        if strand1 == "-":
            edstrs1 = edstrs1[::-1]
            edstrs1 = processSeq.get_reverse_str(edstrs1)
        if strand2 == "-":
            edstrs2 = edstrs2[::-1]
            edstrs2 = processSeq.get_reverse_str(edstrs2) 
        cgm[number_positive] = processSeq.countCG(edstrs3)
        cg1[number_positive] = processSeq.countCG(edstrs1)
        cg2[number_positive] = processSeq.countCG(edstrs2)
        
        if "N" in edstrs1 or "N" in edstrs2:
            table = table.drop(i)
            continue

        outstr = "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(table["label"][i],edstrs1,strand1,edstrs2,strand2,str(chromosome),motif1,motif2)
        label_file.write(outstr)
        number_positive += 1
    table["CG1"] = cg1[:number_positive]
    table["CG2"] = cg2[:number_positive]
    table["CGM"] = cgm[:number_positive]
    print(len(table))
    table.to_csv("./Temp/%s/%s/LabelData_select.csv" %(cell,direction),index=False)

#load word2vec model or train a new one
def getWord_model(word,num_features,min_count):
    word_model = ""
    if not os.path.isfile("./Temp/model"):
        sentence = LineSentence("./Temp/Unsupervised" ,max_sentence_length = 15000)
        print("Start Training Word2Vec model...")
        # Set values for various parameters
        num_features = int(num_features)      # Word vector dimensionality
        min_word_count = int(min_count)      # Minimum word count
        num_workers = 20         # Number of threads to run in parallel
        context = 20            # Context window size
        downsampling = 1e-3     # Downsample setting for frequent words

        # Initialize and train the model
        print("Training Word2Vec model...")
        word_model = Word2Vec(sentence, workers=num_workers,\
                        vector_size=num_features, min_count=min_word_count, \
                        window=context, sample=downsampling, seed=1,epochs = 50)
        word_model.init_sims(replace=False)
        word_model.save("./Temp/model")
        #print word_model.most_similar("CATAGT")
    else:
        print("Loading Word2Vec model...")
        word_model = Word2Vec.load("./Temp/model")
        #word_model.init_sims(replace=True)


    return word_model

def getDNA_split(DNAdata,word):

    DNAlist1 = []
    DNAlist2 = []
    counter = 0
    for DNA in DNAdata["seq1"]:
        if counter % 100 == 0:
            print("DNA %d of %d\r" % (counter, 2*len(DNAdata)))
            sys.stdout.flush()

        DNA = str(DNA).upper()
        DNAlist1.append(processSeq.DNA2Sentence(DNA,word).split(" "))

        counter += 1

    for DNA in DNAdata["seq2"]:
        if counter % 100 == 0:
            print("DNA %d of %d\r" % (counter, 2*len(DNAdata)))
            sys.stdout.flush()

        DNA = str(DNA).upper()
        DNAlist2.append(processSeq.DNA2Sentence(DNA,word).split(" "))

        counter += 1
    return DNAlist1,DNAlist2

def getAvgFeatureVecs(DNAdata1,DNAdata2,model,num_features, word,cell,direction):
    counter = 0
    DNAFeatureVecs = np.zeros((len(DNAdata1),2*num_features), dtype="float32")
    
    for DNA in DNAdata1:
        if counter % 1000 == 0:
            print("DNA %d of %d\r" % (counter, len(DNAdata1)))
            sys.stdout.flush()

        DNAFeatureVecs[counter][0:num_features] = np.mean(model[DNA],axis = 0)
        counter += 1
    
    counter = 0
    for DNA in DNAdata2:
        if counter % 1000 == 0:
            print("DNA %d of %d\r" % (counter, len(DNAdata2)))
            sys.stdout.flush()
        DNAFeatureVecs[counter][num_features:2*num_features] = np.mean(model[DNA],axis = 0)
        counter += 1

    return DNAFeatureVecs

def run(word, num_features,cell,direction):
    warnings.filterwarnings("ignore")

    global word_model,data

    word = int(word)
    num_features = int(num_features)
    word_model=""
    min_count=10

    word_model = getWord_model(word,num_features,min_count)

    # Read data
    data = pd.read_table('./Temp/%s/%s/LabelSeq' %(cell,direction),sep = "\t")
    print("Generating Training and Testing Vector")
    
    
    datawords1,datawords2 = getDNA_split(data,word)
    dataDataVecs = getAvgFeatureVecs(datawords1,datawords2,word_model,num_features,word,cell,direction)

    print(dataDataVecs.shape)
    np.save("./Temp/%s/%s/datavecs.npy" %(cell,direction),dataDataVecs)
    
    '''
    dict1 = genkmerdict(6)
    datawords1,datawords2 = getDNA_split(data,6)
    dataDataVecs = getkmervec(datawords1,datawords2,dict1)
    np.save("./Temp/%s/%s/kmervecs.npy" %(cell,direction),dataDataVecs)
    '''

def run2(word, num_features,cell,direction):
    warnings.filterwarnings("ignore")

    global word_model,data

    word = int(word)
    num_features = int(num_features)
    word_model=""
    min_count=10

    word_model = getWord_model(word,num_features,min_count)
    return word_model

def preprocess_loop_ctcf_mp(cell_line_list):
    '''
    生成环的训练数据
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
    model = run2(6,100,'gm12878','conv')
    for cell_line in cell_line_list:
        pos_total_sample_num = 0
        print('Preprocessing data for cell line: {}'.format(cell_line))
        chr_list = list()
        features_list = list()
        labels_list = list()
        for i in range(1, 6):
            pos_loop_path = POS_LOOP_PATH.format(cell_line, i)+"_v2"
            print("=。=", pos_loop_path)
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
                # DL score
                seq1 = fields[5][233:767] # cut middle 534bp
                seq2 = fields[11][233:767]
                if "N" in seq1 or "N" in seq2: continue
                # seq feature
                seq1 = [seq1[i:i+6] for i in range(0, len(seq1)-5)]
                seq1 = model.wv[seq1]
                seq1 = np.mean(seq1,axis = 0) # shape [100,]
                seq2 = [seq2[i:i+6] for i in range(0, len(seq2)-5)]
                seq2 = model.wv[seq2]
                seq2 = np.mean(seq2,axis = 0) # shape [100,]
                # concat all features
                features = np.concatenate((seq1, seq2, [length], [motif_pattern], [avg_motif_strength], [std_motif_strength], [avg_conservation], [std_conservation], [avg_HS], [std_HS], [HS_in_between], [HS_left], [HS_right]))
                
                chr_list.append(chrom)
                features_list.append(features)
                labels_list.append(response)
        neg_sample_num = pos_total_sample_num // 10
        for typee in range(1, 6):
            for i in range(2):
                neg_loop_path = NEG_LOOP_PATH.format(cell_line, typee, i)+"_v2"
                print("=。=", neg_loop_path)
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
                    # DL score
                    seq1 = fields[5][233:767] # cut middle 534bp
                    seq2 = fields[11][233:767]
                    if "N" in seq1 or "N" in seq2: continue
                    # seq feature
                    seq1 = [seq1[i:i+6] for i in range(0, len(seq1)-5)]
                    seq1 = model.wv[seq1]
                    seq1 = np.mean(seq1,axis = 0) # shape [100,]
                    seq2 = [seq2[i:i+6] for i in range(0, len(seq2)-5)]
                    seq2 = model.wv[seq2]
                    seq2 = np.mean(seq2,axis = 0) # shape [100,]
                    # concat all features
                    features = np.concatenate((seq1, seq2, [length], [motif_pattern], [avg_motif_strength], [std_motif_strength], [avg_conservation], [std_conservation], [avg_HS], [std_HS], [HS_in_between], [HS_left], [HS_right]))
                    chr_list.append(chrom)
                    features_list.append(features)
                    labels_list.append(response)
        chr_list = np.array(chr_list)
        features_list = np.array(features_list, dtype=np.float32)
        labels_list = np.array(labels_list)
        print("chr_list.shape: ", chr_list.shape)
        print("features_list.shape: ", features_list.shape)
        print("labels_list.shape: ", labels_list.shape)
        np.save("./Temp/{}.chr".format(cell_line), chr_list)
        np.save("./Temp/{}.features".format(cell_line), features_list)
        np.save("./Temp/{}.labels".format(cell_line), labels_list)

if __name__ == "__main__":
    # Unsupervised(Range=250, word=6) # 这个是根据CTCF生成的，和cell line 无关，无需重新生成
    # model = run2(6,100,'gm12878','conv') # 这个没用
    preprocess_loop_ctcf_mp(cell_line_list=["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"])

