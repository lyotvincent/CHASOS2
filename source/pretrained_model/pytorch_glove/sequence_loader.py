import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, glob
from parameters import GLOVE_PATH, PRETRAINED_DATA_PATH, CELL_LINE_LIST, POS_LOOP_PATH
from tools import KMerDictionary

def load_loop_seq():
    seqs = list()
    kmd = KMerDictionary()
    with open(POS_LOOP_PATH.format("GM12878", "1"), 'r') as f:
        for line in f:
            fields = line.strip().split()
            seqs.append(kmd.gen_kmers(fields[5]))
            seqs.append(kmd.gen_kmers(fields[11]))

    return seqs

def load_pretrained_data_seq(cell_line=""):
    seqs = list() # this seqs is different from the seqs in load_loop_seq(), this seqs have labels
    file_paths = glob.glob(PRETRAINED_DATA_PATH+f'/{cell_line}*')
    kmd = KMerDictionary()
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            if file_path.endswith('_pos_anchor_ocr.csv'):
                label = 1
            elif file_path.endswith('_pos_anchor_ccr.csv'):
                label = 2
            elif file_path.endswith('_neg_anchor_ocr.csv'):
                label = 3
            elif file_path.endswith('_neg_anchor_ccr.csv'):
                label = 4
            else:
                raise ValueError('file_path is not valid')
            for line in f:
                fields = line.strip().split()
                seqs.append((kmd.gen_kmers(fields[3]), label))

    return seqs

if __name__ == '__main__':
    seqs = load_loop_seq()
    # print(seqs)
    with open(GLOVE_PATH+r'/data/GM12878_corpus.pickle', 'wb') as f:
        pickle.dump(seqs, f)

    # seqs = load_pretrained_data_seq()
    # print(seqs)
    # print(len(seqs))
    # print(len(seqs[0]))
    # print(len(seqs[0][0]))
    # with open(GLOVE_PATH+r'/data/pretrained_data_corpus.pickle', 'wb') as f:
    #     pickle.dump(seqs, f)

    # for cell_line in CELL_LINE_LIST:
    #     seqs = load_pretrained_data_seq(cell_line)
    #     print(len(seqs))
    #     with open(GLOVE_PATH+f'/data/pretrained_{cell_line}_corpus.pickle', 'wb') as f:
    #         pickle.dump(seqs, f)



