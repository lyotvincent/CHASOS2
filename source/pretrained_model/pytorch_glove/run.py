import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import pickle
import torch

from glove import GloVeModel
from tools import KMerDictionary
from parameters import GLOVE_PATH

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MODLE_PATH = GLOVE_PATH+r'/model/glove.pt'
# DOC_PATH = GLOVE_PATH+r'/data/GM12878_corpus.pickle'
DOC_PATH = GLOVE_PATH+r'/data/pretrained_data_corpus.pickle'
COMATRIX_PATH = GLOVE_PATH+r'/data/comat.pickle'
LANG = 'en_core_web_sm'
# GLOVE_EMBEDDING_PATH = GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt' #GM12878_corpus.pickle
GLOVE_EMBEDDING_PATH = GLOVE_PATH+r'/model/glove_embedding.pt' # pretrained_data_corpus.pickle
# EMBEDDING_SIZE = 128
EMBEDDING_SIZE = 16
CONTEXT_SIZE = 3
NUM_EPOCH = 100
BATHC_SIZE = 512
LEARNING_RATE = 0.01


def preprocess():
    """ Get corpus and vocab_size from raw text

    Args:
        file_path (str): raw file path

    Returns:
        corpus (list): list of idx words
        vocab_size (int): vocabulary size
    """

    ## for GM12878_corpus.pickle
    # with open(DOC_PATH, 'rb') as fp:
    #     doc = pickle.load(fp)
    # print(torch.tensor(doc).shape)
    # corpus = doc
    # dictionary = KMerDictionary()
    # corpus = dictionary.handle_corpus(doc)

    ## for pretrained_data_corpus.pickle
    with open(DOC_PATH, 'rb') as fp:
        doc = pickle.load(fp)
    corpus = [seq for seq, _ in doc]

    vocab_size = 4**5
    return corpus, vocab_size


def train_glove_model():
    # preprocess
    corpus, vocab_size = preprocess()
    # print(len(corpus))

    # specify device type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init vector model
    logging.info("init model hyperparameter")
    model = GloVeModel(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
    model.to(device)

    # fit corpus to count cooccurance matrix
    model.fit(corpus)

    cooccurance_matrix = model.get_coocurrance_matrix()
    # saving cooccurance_matrix
    with open(COMATRIX_PATH, mode='wb') as fp:
        pickle.dump(cooccurance_matrix, fp)

    model.train(NUM_EPOCH, device, learning_rate=LEARNING_RATE)

    # save model for evaluation
    model_dict = model.state_dict()
    # torch.save(model_dict, MODLE_PATH)
    torch.save({'focal_embeddings.weight': model_dict['_focal_embeddings.weight']}, GLOVE_EMBEDDING_PATH)

if __name__ == '__main__':
    train_glove_model()
