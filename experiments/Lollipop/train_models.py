'''
train program in py3 for Lollipop, come from Lollipop's train_models.py
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

    # Use Random Forest classifier to predict interactions
    random_state = np.random.RandomState(0)
    n_estimator = 100
    rf = RandomForestClassifier(max_depth = 36,max_features=18, n_estimators=n_estimator, random_state = random_state,n_jobs=-1, class_weight={0:1,1:n2p}) 

    print('Training model...') # 0.9094202898550725
    EPOCH = 20
    for i in range(EPOCH):
        rf.fit(train_X, train_Y)
        r_score = rf.score(val_X, val_Y)
        print(i, r_score)


if __name__ == "__main__":
    main()
