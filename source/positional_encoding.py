'''
@author: 孙嘉良
'''

import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, dropout=0., max_seq_len=5000):
        '''
        @param max_seq_len: the max tokens number in a sequence
        @param d_model: the embedding dimension of a token
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe.size())
        # print(x.size())
        # print(self.pe[:, :x.size(1), :x.size(2)].size())
        try:
            x = x + self.pe[:, :x.size(1), :x.size(2)] # type: ignore
        except Exception as e:
            print(e)
            print(x)
        return self.dropout(x)

# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=128)
#         self.pos_encoding = PositionalEncoding(d_model=128, max_seq_len=100)

#         self.fc = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         x = x.mean(dim=1)
#         x = self.fc(x)
#         return x

# if __name__ == '__main__':
#     # 创建一个包含10个单词的序列，每个单词嵌入到长度为20的向量中
#     x = torch.zeros(10, 20, 30)
#     # 创建一个positional encoding层，输入维度为20
#     pos_enc = PositionalEncoding(30)
#     # 将positional encoding层应用到输入序列中
#     # print(x)
#     output = pos_enc(x)
#     print(output.shape)  # 输出 torch.Size([10, 20])
#     # print(output)


