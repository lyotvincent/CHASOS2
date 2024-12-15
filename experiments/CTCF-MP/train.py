#encoding:utf-8
import pandas as pd
import numpy as np
import os
import sys
import math
import random
import processSeq
import warnings

from sklearn import preprocessing
import sklearn.preprocessing
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
# from sklearn.cross_validation import cross_val_score,StratifiedKFold,train_test_split

import xgboost as xgb
import joblib

def train():

    # # bychr
    # VAL_CHR_LIST = ["chr22", "chrX"]
    # cell_line_list = ["AG04450", "BJ", "Caco-2", "GM12878", "HCT116", "IMR-90", "K562", "MCF-7"]
    # chr_list = np.load("./Temp/{}.chr.npy".format("AG04450"))
    # features_list = np.load("./Temp/{}.features.npy".format("AG04450"))
    # labels_list = np.load("./Temp/{}.labels.npy".format("AG04450"))
    # for cell_line in cell_line_list[1:]:
    #     chr_list = np.concatenate((chr_list, np.load("./Temp/{}.chr.npy".format(cell_line))), axis=0)
    #     features_list = np.concatenate((features_list, np.load("./Temp/{}.features.npy".format(cell_line))), axis=0)
    #     labels_list = np.concatenate((labels_list, np.load("./Temp/{}.labels.npy".format(cell_line))), axis=0)

    # print("chr_list.shape: ", chr_list.shape)
    # print("features_list.shape: ", features_list.shape)
    # print("labels_list.shape: ", labels_list.shape)
    # for i in range(len(chr_list)):
    #     if labels_list[i] not in [0, 1]:
    #         print(i, chr_list[i], labels_list[i])
    #         break

    # val_indics = [i in VAL_CHR_LIST for i in chr_list]
    # train_indics = [not i for i in val_indics]
    # train_features = features_list[train_indics]
    # train_labels = labels_list[train_indics]
    # val_features = features_list[val_indics]
    # val_labels = labels_list[val_indics]

    # byhealth
    feat_AG04450 = np.load("./Temp/{}.features.npy".format("AG04450"))
    labels_AG04450 = np.load("./Temp/{}.labels.npy".format("AG04450"))
    feat_BJ = np.load("./Temp/{}.features.npy".format("BJ"))
    labels_BJ = np.load("./Temp/{}.labels.npy".format("BJ"))
    feat_GM12878 = np.load("./Temp/{}.features.npy".format("IMR-90"))
    labels_GM12878 = np.load("./Temp/{}.labels.npy".format("IMR-90"))
    feat_IMR90 = np.load("./Temp/{}.features.npy".format("GM12878"))
    labels_IMR90 = np.load("./Temp/{}.labels.npy".format("GM12878"))
    train_features = np.concatenate((feat_AG04450, feat_BJ, feat_GM12878, feat_IMR90), axis=0)
    train_labels = np.concatenate((labels_AG04450, labels_BJ, labels_GM12878, labels_IMR90), axis=0)
    # cancer: ["Caco-2", "HCT116", "K562", "MCF-7"]
    feat_Caco2 = np.load("./Temp/{}.features.npy".format("Caco-2"))
    labels_Caco2 = np.load("./Temp/{}.labels.npy".format("Caco-2"))
    feat_HCT116 = np.load("./Temp/{}.features.npy".format("HCT116"))
    labels_HCT116 = np.load("./Temp/{}.labels.npy".format("HCT116"))
    feat_K562 = np.load("./Temp/{}.features.npy".format("K562"))
    labels_K562 = np.load("./Temp/{}.labels.npy".format("K562"))
    feat_MCF7 = np.load("./Temp/{}.features.npy".format("MCF-7"))
    labels_MCF7 = np.load("./Temp/{}.labels.npy".format("MCF-7"))
    val_features = np.concatenate((feat_Caco2, feat_HCT116, feat_K562, feat_MCF7), axis=0)
    val_labels = np.concatenate((labels_Caco2, labels_HCT116, labels_K562, labels_MCF7), axis=0)

    print("train_features.shape: ", train_features.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("val_features.shape: ", val_features.shape)
    print("val_labels.shape: ", val_labels.shape)

    model = xgb.XGBClassifier(max_depth=12,learning_rate=0.01, n_estimators = 500,nthread=30,tree_method='gpu_hist', gpu_id=0)

    EPOCH = 10
    best_epoch = 0
    best_score = 0
    for i in range(EPOCH):
        model.fit(train_features, train_labels)
        r_score = model.score(val_features, val_labels)
        if r_score > best_score:
            best_epoch = i
            best_score = r_score
            #lr是一个LogisticRegression模型
            joblib.dump(model, './Temp/XGBClassifier.joblib')
        print(i, r_score)
    print('best score: ', best_epoch, best_score)
    # print(model.feature_importances_)W
    best_model = joblib.load('./Temp/XGBClassifier.joblib')
    r_score = best_model.score(val_features, val_labels)
    print('best score2: ', r_score)
    predictions = best_model.predict_proba(val_features)
    predictions = predictions[:, 1]
    roc_auc = roc_auc_score(val_labels, predictions)
    prc_auc = average_precision_score(val_labels, predictions)
    print('roc_auc: ', roc_auc)
    print('prc_auc: ', prc_auc)


if __name__=="__main__":
    train()


