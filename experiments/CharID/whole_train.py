'''
train program in py3 for Lollipop, come from Lollipop's train_models.py
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from source.parameters import ROOT_DIR, CELL_LINE_LIST

LOLLIPOP_PATH = ROOT_DIR+"/experiments/Lollipop/"

def main():

    print('Reading in features...')
    data = pd.read_table(LOLLIPOP_PATH+"/training_data/GM12878.pos_loop", sep='\t')
    # features = data.columns.values[4:]
    # print(features.shape)

    # Evaluating whether the data are balanced..
    num_pos = data[data['response'] == 1].shape[0]
    num_neg = data[data['response'] == 0].shape[0]
    n2p = float(num_neg)/float(num_pos)
    print('Negative samples were '+str(n2p)+' times more than the positive loops...')

    val_chr_list = ["chr22", "chrX"]
    train_data = data[data['chrom'].isin(val_chr_list) == False]
    val_data = data[data['chrom'].isin(val_chr_list) == True]
    print(f"train_data.shape: {train_data.shape}")
    print(f"val_data.shape: {val_data.shape}")
    train_X = train_data.iloc[:,4:]
    train_Y = np.array(train_data['response'])
    val_X = val_data.iloc[:,4:]
    val_Y = np.array(val_data['response'])

    gbm0 = GradientBoostingClassifier(n_estimators=235, random_state=10)
    gbm0.fit(train_X,train_Y)

    print('Training model...') # 0.9202898550724637
    EPOCH = 10
    for i in range(EPOCH):
        gbm0.fit(train_X, train_Y)
        r_score = gbm0.score(val_X, val_Y)
        print(i, r_score)


if __name__ == "__main__":
    main()
