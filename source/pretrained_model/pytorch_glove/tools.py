'''
@author: 孙嘉良
'''

class KMerDictionary:
    """
    @purpose: a class for building dictionary for k-mer -> index
    """

    def __init__(self, kmer_len=5) -> None:
        self.kmer_len = kmer_len
        self.__gen_dict()

    def __gen_dict(self):
        base_list = ['A', 'T', 'C', 'G']
        kmer_list = ['A', 'T', 'C', 'G']
        for _ in range(self.kmer_len-1):
            temp_list = list()
            for kmer in kmer_list:
                for base in base_list:
                    temp_list.append(kmer + base)
            kmer_list = temp_list
        self.kmer_dict = {kmer: i for i, kmer in enumerate(kmer_list)}

    def gen_kmers(self, seq: str, stride=1):
        kmers = list()
        for i in range(0, len(seq)-self.kmer_len+1, stride):
            kmers.append(seq[i:i+self.kmer_len])
        return [self.kmer_dict[kmer] for kmer in kmers]

    def handle_corpus(self, seqs):
        corpus = list()
        for seq in seqs:
            for kmer in seq:
                corpus.append(self.kmer_dict[kmer])
        return corpus


if __name__ == '__main__':
    kmd = KMerDictionary()
    # print(kmd.kmer_dict)
    kmers = kmd.gen_kmers("ATCGAGTCGAGT")
    print(kmers)

