import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from tools import KMerDictionary
from parameters import GLOVE_PATH

# 定义模型
embed = nn.Embedding(num_embeddings=4**5, embedding_dim=128).type(torch.float64)

# print(torch.load('./model/glove.pt'))
# print(torch.load('./model/glove.pt')['_focal_embeddings.weight'])

# 加载模型参数
# model.load_state_dict(torch.load('./model/glove_embedding.pt'))

saved_dict = torch.load(GLOVE_PATH+r'/model/glove_embedding.pt')
embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
model_dict = embed.state_dict()
model_dict.update(embedding_dict)
embed.load_state_dict(model_dict)

# 将模型设置为评估模式
embed.eval()

seqs = [["ATCAG", "TGACT", "TATGC", "CATGT", "CTGAT", "GCATT", "ATTGC", "TATGC"]]
seqs = [["ATCAG", "TGACT"], ["TATGC", "CATGT"], ["CTGAT", "GCATT"], ["ATTGC", "TATGC"]]
kmd = KMerDictionary()
embed_tensor = kmd.handle_corpus(seqs)
embed_tensor = torch.tensor(embed_tensor)
print(embed_tensor)
result = embed(embed_tensor)
print(result.size())


