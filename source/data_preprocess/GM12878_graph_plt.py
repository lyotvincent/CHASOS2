'''
画GM12878的环的图
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
# ! networkx的v2.8.4配合matplotlib的v3.6.2会报错
# ! networkx的v2.8.4配合scipy的v1.7会报错
# networkx的v2.8.4, matplotlib的v3.5.2, scipy v1.8.1能运行
import networkx as nx # v2.8.4
import matplotlib.pyplot as plt # 3.5.2
from parameters import *
from utils import TimeStatistic

def load_GM12878_loop():
    GM12878_path = CHIA_PET2_RESULT_PATH.format("GM12878")+"/GM12878.significant.interactions.intra.filtered.MICC"
    f = open(GM12878_path, "r")
    lines = f.readlines()
    f.close()

    chr2color = {'chr1': '#FFEE6F', 'chr2': '#D3A237', 'chr3': '#EA5514', 'chr4': '#F8C6B5',\
                 'chr5': '#F29A76', 'chr6': '#DD6B4F', 'chr7': '#CE93BF', 'chr8': '#F091A0',\
                 'chr9': '#DA6D83', 'chr10': '#A73766', 'chr11': '#8F1D22', 'chr12': '#7C4449',\
                 'chr13': '#D5EBE1', 'chr14': '#BFD1B2', 'chr15': '#C0D09D', 'chr16': '#75C1C4',\
                 'chr17': '#178069', 'chr18': '#2B2E77', 'chr19': '#EAEEF1', 'chr20': '#E5E3E5',\
                 'chr21': '#EEEAD9', 'chr22': '#99806C', 'chrX': '#6B5458', 'chrY': '#344B5C'}
    edge_color_list = list()
    edge_list = list()
    edge_chr_list = list()
    node2order = dict()
    for line in lines:
        fields = line.strip().split()
        if fields[13] == '.' or fields[14] == '.': continue
        edge_chr_list.append(fields[0])
        edge_color_list.append(chr2color[fields[0]]) # field[0] example: chr1
        node_u, node_v = fields[6], fields[7] # example: peak_3 peak_8
        if node_u not in node2order.keys():
            node2order[node_u] = len(node2order)
        if node_v not in node2order.keys():
            node2order[node_v] = len(node2order)
        edge_list.append((node2order[node_u], node2order[node_v]))
    print(f"total {len(node2order)} nodes, examples: {list(node2order.values())[:5]}")
    print(f"total {len(edge_list)} edges, examples: {edge_list[:5]}")
    return node2order, edge_list, edge_chr_list, edge_color_list

@TimeStatistic
def draw_GM12878_loop():
    G=nx.Graph()
    # G.add_nodes_from(np.arange(7))
    # G.add_edges_from([(1,2),(1,3),(2,4),(2,5)])
    node2order, edge_list, edge_chr_list, edge_color_list = load_GM12878_loop()
    G.add_nodes_from(np.arange(len(node2order)))
    print(f"add_nodes complete, total {len(G.nodes)} edges")
    G.add_edges_from(edge_list)
    print(f"add_edges complete, total {len(G.edges)} edges")
    for i, (u, v) in enumerate(edge_list):
        G[u][v]["chr"] = edge_chr_list[i]
    print(f"total {len(G.edges)} edges, examples: {list(G.edges(data=True))[:5]}")
    plt.figure(dpi=300, figsize=(16, 8))
    pos = nx.spring_layout(G,scale=2) # spring_layout
    nx.draw(G=G, pos=pos, node_size=0.05, node_color='#000000', alpha=1, width=0.5, edge_color=edge_color_list)
    print("draw complete")
    # plt.show()
    plt.savefig(fname=ROOT_DIR+r"/figure/GM12878_loop.svg", format="svg", bbox_inches="tight")
    # plt.savefig(fname=ROOT_DIR+r"/figure/GM12878_loop.png", format="png", bbox_inches="tight")
    # plt.savefig(fname=ROOT_DIR+r"/figure/GM12878_loop.eps", format="eps", bbox_inches="tight")

if __name__ == '__main__':
    # load_GM12878_loop()
    draw_GM12878_loop()
