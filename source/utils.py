import time, math, random
from functools import wraps
from parameters import *
import matplotlib.pyplot as plt
import seaborn as sns

class TimeStatistic():
    '''
    a decorator for record time and count
    '''
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            print(f"[INFO] {'>'*7} TimeStatistic Started {'>'*7}\tfunction name: {self.func.__name__}\t❛‿˂̵✧")
            start_time = time.perf_counter()
            result = self.func(*args, **kwargs)
            end_time = time.perf_counter()
            print(f"[INFO] {'<'*7} TimeStatistic  Ended  {'<'*7}\tfunction name: {self.func.__name__}, runtime: {end_time-start_time}s")
            return result
        return wrapper(*args, **kwargs)


def count_max_min_loop_length():
    '''
    @purpose: 这个函数原本是想统计MICC中染色质环的长度的，但是结果是[2436, 238111832]。
            但在关于染色质环长度的研究中，
            convergent:1 tandem:2 divergent:3。加条件筛选了仅这三种环，结果没变化多少[2631, 214822565]。
    '''
    min_loop_length, max_loop_length = math.inf, -math.inf
    loop_length_list = list()
    for cell_line in CELL_LINE_LIST:
        print(f"[INFO] Preprocessing {cell_line} loop")
        chia_pet2_result_path = CHIA_PET2_RESULT_PATH.format(cell_line)+'/{}.significant.interactions.intra.filtered.MICC'.format(cell_line)
        f = open(chia_pet2_result_path)
        for line in f:
            fields = line.strip().split()
            anchor1_motifs_str, anchor2_motifs_str = fields[13:15]
            if "." in anchor1_motifs_str or ";" in anchor1_motifs_str or "." in anchor2_motifs_str or ";" in anchor2_motifs_str:
                continue
            loop_length = int(fields[5]) - int(fields[1])
            loop_length_list.append(loop_length)
            if loop_length < min_loop_length:
                min_loop_length = loop_length
            if loop_length > max_loop_length:
                max_loop_length = loop_length
        f.close()
    print(f"[INFO] Loop Length: [{min_loop_length}, {max_loop_length}]")

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
    sns.violinplot(data=loop_length_list)
    plt.ylim(0, 1000000)
    # plt.tight_layout()
    plt.savefig(fname=ROOT_DIR+r"/figure/pos_loop_size_statistics.svg", format="svg", bbox_inches="tight")

def get_mid_point(start, end):
    '''
    @purpose: get mean of start end, the output maybe 整数或者小数部分为0.5的浮点数
    '''
    return (start+end) / 2

def shuffle_a_sequence(seq):
    seq = list(seq)
    random.shuffle(seq)
    return "".join(seq)

if __name__ == '__main__':
    count_max_min_loop_length()

