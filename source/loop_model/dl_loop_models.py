
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import itertools
import joblib
import xgboost as xgb
import warnings
from parameters import GLOVE_PATH, FINE_TUNED_OCR_MODEL_PATH, LOOP_MODEL_PATH
from positional_encoding import PositionalEncoding
from pretrained_model.custom_layer import *
from pretrained_model import models as anchor_models
from fine_tuned_ocr_model import ocr_models
class DL_Loop_Model_v20230530(nn.Module):
    """
    @description:
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.anchor1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.anchor2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.kmer_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(336, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

        self.anchor_score_model = anchor_models.AnchorModel_v20230516_v1()
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_05_16_12_57_12_GM12878_AnchorModel_v20230516_v1.pt')
        saved_dict = checkpoint['model']
        self.anchor_score_model.load_state_dict(saved_dict)
        self.ocr_score_model = ocr_models.OCRModel_v20230524_v1()
        checkpoint = torch.load(FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_05_26_21_59_09_GM12878_OCR_ext2_OCRModel_v20230524_v1.pt')
        saved_dict = checkpoint['model']
        self.ocr_score_model.load_state_dict(saved_dict)


    def forward(self, features, kmer_data_1, kmer_data_2):
        # features shape: [batch_size, 6]
        # kmer_data shape: [batch_size, 996]
        h1 = self.embedding_1(kmer_data_1)
        h1 = self.pos_encoding(h1)                    # output shape: (batch_size, 996, 16)
        h2 = self.embedding_1(kmer_data_2)
        h2 = self.pos_encoding(h2)                    # output shape: (batch_size, 996, 16)

        h1 = h1.transpose(1, 2)
        h1 = self.slice_layer_0(h1)                  # output shape: (batch_size, 64, 512)
        h2 = h2.transpose(1, 2)
        h2 = self.slice_layer_0(h2)                  # output shape: (batch_size, 64, 512)

        h1 = self.anchor1(h1)                  # output shape: (batch_size, 256, 128)
        h2 = self.anchor2(h2)                  # output shape: (batch_size, 256, 128)
        h = torch.cat([h1, h2], dim=2)         # output shape: (batch_size, 256, 256)
        h = self.kmer_layer(h)                      # output shape: (batch_size, 32)

        anchor1_score = self.anchor_score_model(kmer_data_1)        # output shape: (batch_size, 1)
        anchor2_score = self.anchor_score_model(kmer_data_2)        # output shape: (batch_size, 1)
        ocr1_score = self.ocr_score_model(kmer_data_1)       # output shape: (batch_size, 1)
        ocr1_score = nn.AdaptiveAvgPool1d(100)(ocr1_score)
        ocr2_score = self.ocr_score_model(kmer_data_2)       # output shape: (batch_size, 1)
        ocr2_score = nn.AdaptiveAvgPool1d(100)(ocr2_score)
        h = torch.cat([h, features, anchor1_score, anchor2_score, ocr1_score, ocr2_score], dim=1)         # output shape: (batch_size, 38)
        h = self.out_layer(h)                       # output shape: (batch_size, 1)
        return h

class ML_Loop_Model_v20230531(nn.Module):
    """
    @description:
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        self.onehot_layer = OneHotConcatModule()
        BRANCH_NUM = 8
        MAX_FEATURES = 3
        FEATURE_INDEX = [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17]]
        FEATURE_INDEX_SET = set(range(18))
        self.ensemble_module_list = nn.ModuleList()
        # self.reverse_ensemble_module_list = nn.ModuleList()
        # self.difference_module_list = nn.ModuleList()
        self.feature_combination = []
        # for i in range(len(FEATURE_INDEX)):
        #     for j in range(i+1, len(FEATURE_INDEX)):
        #         for k in range(j+1, len(FEATURE_INDEX)):
        #             self.feature_combination.append(FEATURE_INDEX[i]+FEATURE_INDEX[j]+FEATURE_INDEX[k])
        self.feature_combination = list(itertools.combinations(FEATURE_INDEX, 3))
        temp_comb = list()
        for comb1 in self.feature_combination:
            temp = list()
            for single in comb1:
                temp += single
            temp_comb.append(temp)
        self.feature_combination = temp_comb
        ## 在C_3_14中共有364个组合，其中至少包含前三个特征之一的组合数量为C_3_14 - C_3_11 = 364 - 165 = 199
        self.feature_combination = [i for i in self.feature_combination if set(i).intersection(set([0,1,2,3,4,5,6]))]
        # self.reverse_feature_combination = [list(FEATURE_INDEX_SET.difference(set(i))) for i in self.feature_combination]

        # for i in range(3):
        #     for i in range(2, 2*(BRANCH_NUM+1), 2):
        #         self.ensemble_module_list.append(self.__append_branch(i, 18))

        for f_index in self.feature_combination:
            self.ensemble_module_list.append(append_branch(depth=4, feature_num=len(f_index)))
        print("MODEL ensemble_module_list len: ", len(self.ensemble_module_list))
        # for f_index in self.reverse_feature_combination:
        #     self.reverse_ensemble_module_list.append(self.__append_branch(depth=4, feature_num=len(f_index)))
        # print("MODEL reverse_ensemble_module_list len: ", len(self.reverse_ensemble_module_list))
        # for _ in range(len(self.feature_combination)):
        #     self.difference_module_list.append(nn.Sequential(
        #         nn.Linear(2, 2),
        #         nn.ReLU(),
        #         nn.Linear(2, 2),
        #         nn.ReLU(),
        #         nn.Linear(2, 1),
        #         nn.Sigmoid()
        #     ))


        self.gbc_model = joblib.load(LOOP_MODEL_PATH+'/model/gradientboostingclassifier.joblib')
        self.trees = self.gbc_model.estimators_
        # self.xgb_model = joblib.load(LOOP_MODEL_PATH+'/model/xgbclassifier.joblib')
        # self.booster = self.xgb_model.get_booster()
        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(len(self.ensemble_module_list)+len(self.trees)+self.booster.num_boosted_rounds()+1, 16),
            nn.Linear(len(self.ensemble_module_list)+len(self.trees)+1, 16),
            nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.step_counter = 0

    def forward(self, features):
        features_18 = self.onehot_layer(features)
        dl_outputs = torch.Tensor().cuda()
        # reverse_outputs = torch.Tensor().cuda()
        # difference_outputs = torch.Tensor().cuda()
        for i, branch in enumerate(self.ensemble_module_list):
            output = branch(features_18[:, self.feature_combination[i]])
            dl_outputs = torch.cat((dl_outputs, output), dim=-1)

        self.step_counter += 1
        if self.step_counter == 87+1:
            print(f"ML model is used. step_counter=={self.step_counter}")
        if self.step_counter > 87:
            cpu_features = features.cpu().detach().numpy()
            ml_outputs = self.gbc_model.predict(cpu_features)
            ml_outputs = torch.from_numpy(ml_outputs).cuda().reshape(-1, 1)
            for i in range(len(self.trees)):
                t = self.trees[i][0].predict(cpu_features)
                t = torch.from_numpy(t).cuda().reshape(-1, 1)
                ml_outputs = torch.cat((ml_outputs, t), dim=1)
            # for tree_dump in self.booster:
            #     dtrain = xgb.DMatrix(cpu_features, feature_names=['length', 'motif_pattern', 'avg_motif_strength', 'std_motif_strength', 'avg_conservation', 'std_conservation', 'avg_anchor_score', 'std_anchor_score', 'anchor_score1', 'anchor_score2', 'avg_ocr_score', 'std_ocr_score', 'ocr_score1', 'ocr_score2']) # feature_names=val_X.columns
            #     pred_Y = tree_dump.predict(dtrain)
            #     pred_Y = torch.from_numpy(pred_Y).cuda().reshape(-1, 1)
            #     ml_outputs = torch.cat((ml_outputs, pred_Y), dim=1)
        else:
            ml_outputs = torch.zeros(dl_outputs.shape[0], len(self.trees)+1).cuda()
        outputs = torch.cat((dl_outputs, ml_outputs), dim=1).float()

        # for i, branch in enumerate(self.reverse_ensemble_module_list):
        #     output = branch(features[:, self.reverse_feature_combination[i]])
        #     reverse_outputs = torch.cat((reverse_outputs, output), dim=-1)
        # for i in range(len(self.difference_module_list)):
        #     difference_outputs = torch.cat((difference_outputs, self.difference_module_list[i](torch.cat([outputs[:, i].unsqueeze(1), reverse_outputs[:, i].unsqueeze(1)], dim=1))), dim=-1)
        # outputs = torch.cat([outputs, reverse_outputs, difference_outputs], dim=-1)
        h = self.out_layer(outputs)
        return h, dl_outputs

class OneHotConcatModule(nn.Module):
    def __init__(self, x_dim=1, num_classes=5):
        super(OneHotConcatModule, self).__init__()
        self.x_dim = x_dim
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        feature_dim = x.size(1)

        # 检查x_dim是否合法
        if self.x_dim < 0 or self.x_dim >= feature_dim:
            raise ValueError(f"Invalid x_dim. Must be in the range [0, {feature_dim - 1}]")

        # 获取指定维度上的特征
        selected_feature = x[:, self.x_dim].long()

        # 将特征转换为ONE-HOT编码
        one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        one_hot.scatter_(dim=1, index=selected_feature.unsqueeze(1), value=1.)

        # 将ONE-HOT编码拼接到输入上
        x_concat = torch.cat([one_hot, x], dim=1)

        return x_concat

def append_branch(depth, feature_num):
    _branch = nn.Sequential()
    mid = depth/2
    _branch.add_module('bn_0', nn.BatchNorm1d(feature_num))
    _branch.add_module('linear_0', nn.Linear(feature_num, 32))
    _branch.add_module('relu_0', nn.ReLU())
    for i in range(depth):
        if i == 0 or i == depth-1: continue
        if i < mid:
            _branch.add_module(f'linear_{i}', nn.Linear(16*2**(i), 16*2**(i+1)))
            _branch.add_module(f'bn_{i}', nn.BatchNorm1d(16*2**(i+1)))
            _branch.add_module(f'relu_{i}', nn.ReLU())
        else:
            _branch.add_module(f'linear_{i}', nn.Linear(16*2**(depth-i), 16*2**(depth-i-1)))
            _branch.add_module(f'bn_{i}', nn.BatchNorm1d(16*2**(depth-i-1)))
            _branch.add_module(f'relu_{i}', nn.ReLU())
    _branch.add_module(f'gru_0', nn.GRU(32, 16, num_layers=1, batch_first=True, dropout=0., bidirectional=True))
    _branch.add_module(f'select_0', SelectItem(0))
    _branch.add_module(f'linear_{depth-1}', nn.Linear(32, 1))
    _branch.add_module(f'sigmoid_{depth-1}', nn.Sigmoid())
    return _branch

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class DL_Loop_Model_v20230613(nn.Module):
    """
    @description:
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.anchor1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.anchor2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.kmer_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU()
        )
        
        self.onehot_layer = OneHotConcatModule()
        FEATURE_INDEX = [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17]]
        self.ensemble_module_list = nn.ModuleList()
        self.feature_combination = []
        self.feature_combination = list(itertools.combinations(FEATURE_INDEX, 3))
        temp_comb = list()
        for comb1 in self.feature_combination:
            temp = list()
            for single in comb1:
                temp += single
            temp_comb.append(temp)
        self.feature_combination = temp_comb
        ## 在C_3_14中共有364个组合，其中至少包含前三个特征之一的组合数量为C_3_14 - C_3_11 = 364 - 165 = 199
        self.feature_combination = [i for i in self.feature_combination if set(i).intersection(set([0,1,2,3,4,5,6]))]
        for f_index in self.feature_combination:
            self.ensemble_module_list.append(append_branch(depth=4, feature_num=len(f_index)))
        print("MODEL ensemble_module_list len: ", len(self.ensemble_module_list))

        self.gbc_model = joblib.load(LOOP_MODEL_PATH+'/model/gradientboostingclassifier.joblib')
        self.trees = self.gbc_model.estimators_

        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(len(self.ensemble_module_list)+len(self.trees)+1, 16),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        ## load embedding parameters
        # saved_dict = torch.load(GLOVE_PATH+r'/model/GM12878_loop_glove_embedding.pt')
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

        self.anchor_score_model = anchor_models.AnchorModel_v20230516_v1()
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/model/PretrainedModel_2023_05_16_12_57_12_GM12878_AnchorModel_v20230516_v1.pt')
        saved_dict = checkpoint['model']
        self.anchor_score_model.load_state_dict(saved_dict)
        # self.anchor_score_model.eval()
        # self.anchor_score_model.requires_grad_(False)
        self.ocr_score_model = ocr_models.OCRModel_v20230524_v1()
        checkpoint = torch.load(FINE_TUNED_OCR_MODEL_PATH+r'/model/OCRModel_val_loss_2023_05_26_21_59_09_GM12878_OCR_ext2_OCRModel_v20230524_v1.pt')
        saved_dict = checkpoint['model']
        self.ocr_score_model.load_state_dict(saved_dict)

        self.step_counter = 0

    def forward(self, features, kmer_data_1, kmer_data_2):
        '''
        features shape: [batch_size, 6]
        kmer_data shape: [batch_size, 996]
        '''
        # h1 = self.embedding_1(kmer_data_1)
        # h1 = self.pos_encoding(h1)                    # output shape: (batch_size, 996, 16)
        # h2 = self.embedding_1(kmer_data_2)
        # h2 = self.pos_encoding(h2)                    # output shape: (batch_size, 996, 16)

        # h1 = h1.transpose(1, 2)
        # h1 = self.slice_layer_0(h1)                  # output shape: (batch_size, 64, 512)
        # h2 = h2.transpose(1, 2)
        # h2 = self.slice_layer_0(h2)                  # output shape: (batch_size, 64, 512)

        # h1 = self.anchor1(h1)                  # output shape: (batch_size, 256, 128)
        # h2 = self.anchor2(h2)                  # output shape: (batch_size, 256, 128)
        # h = torch.cat([h1, h2], dim=2)         # output shape: (batch_size, 256, 256)
        # h = self.kmer_layer(h)                      # output shape: (batch_size, 32)

        anchor1_score = self.anchor_score_model(kmer_data_1)        # output shape: (batch_size, 1)
        anchor2_score = self.anchor_score_model(kmer_data_2)        # output shape: (batch_size, 1)
        avg_anchor_score = (anchor1_score + anchor2_score) / 2
        std_anchor_score = torch.std(torch.cat([anchor1_score, anchor2_score], dim=1), dim=1).unsqueeze(1)
        ocr1_score = nn.AdaptiveAvgPool1d(1)(self.ocr_score_model(kmer_data_1))       # output shape: (batch_size, 1)
        ocr1_score = ocr1_score
        ocr2_score = nn.AdaptiveAvgPool1d(1)(self.ocr_score_model(kmer_data_2))        # output shape: (batch_size, 1)
        ocr2_score = ocr2_score
        avg_ocr_score = (ocr1_score + ocr2_score) / 2
        std_ocr_score = torch.std(torch.cat([ocr1_score, ocr2_score], dim=1), dim=1).unsqueeze(1)

        features = torch.cat([features, avg_anchor_score, std_anchor_score, anchor1_score, anchor2_score, avg_ocr_score, std_ocr_score, ocr1_score, ocr2_score], dim=1)

        features_18 = self.onehot_layer(features)
        dl_outputs = torch.Tensor().cuda()
        # reverse_outputs = torch.Tensor().cuda()
        # difference_outputs = torch.Tensor().cuda()
        for i, branch in enumerate(self.ensemble_module_list):
            output = branch(features_18[:, self.feature_combination[i]])
            dl_outputs = torch.cat((dl_outputs, output), dim=-1)

        self.step_counter += 1
        if self.step_counter == 87+1:
            print(f"ML model is used. step_counter=={self.step_counter}")
        if self.step_counter > 87:
            cpu_features = features.cpu().detach().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml_outputs = self.gbc_model.predict(cpu_features)
            ml_outputs = torch.from_numpy(ml_outputs).cuda().reshape(-1, 1)
            for i in range(len(self.trees)):
                t = self.trees[i][0].predict(cpu_features)
                t = torch.from_numpy(t).cuda().reshape(-1, 1)
                ml_outputs = torch.cat((ml_outputs, t), dim=1)
        else:
            ml_outputs = torch.zeros(dl_outputs.shape[0], len(self.trees)+1).cuda()
        pred_result = torch.cat([dl_outputs, ml_outputs], dim=1).float()

        pred_result = self.out_layer(pred_result)                       # output shape: (batch_size, 1)
        return pred_result, dl_outputs

class DL_Loop_Model_v20230614(nn.Module):
    """
    @description: 与0613的区别是，这个模型把anchor score和ocr score在外面计算的。
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.anchor1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.anchor2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=16, e_1_channels=16, e_3_channels=32, e_5_channels=16),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            SliceLayer1D(slice_len=256, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            SliceLayer1D(slice_len=128, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            SliceLayer1D(slice_len=64, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=512, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            SliceLayer1D(slice_len=32, slice_num=4),
            LargeKernelFireBlock1D_v1(input_channels=1024, squeeze_channels=256, e_1_channels=256, e_3_channels=512, e_5_channels=256),
        )
        self.kmer_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        self.onehot_layer = OneHotConcatModule()
        FEATURE_INDEX = [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17]]
        self.ensemble_module_list = nn.ModuleList()
        self.feature_combination = []
        self.feature_combination = list(itertools.combinations(FEATURE_INDEX, 3))
        temp_comb = list()
        for comb1 in self.feature_combination:
            temp = list()
            for single in comb1:
                temp += single
            temp_comb.append(temp)
        self.feature_combination = temp_comb
        ## 在C_3_14中共有364个组合，其中至少包含前三个特征之一的组合数量为C_3_14 - C_3_11 = 364 - 165 = 199
        self.feature_combination = [i for i in self.feature_combination if set(i).intersection(set([0,1,2,3,4,5,6]))]
        for f_index in self.feature_combination:
            self.ensemble_module_list.append(append_branch(depth=4, feature_num=len(f_index)))
        print("MODEL ensemble_module_list len: ", len(self.ensemble_module_list))

        self.gbc_model = joblib.load(LOOP_MODEL_PATH+'/model/gradientboostingclassifier.joblib')
        self.trees = self.gbc_model.estimators_

        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128+len(self.ensemble_module_list)+len(self.trees)+1, 16),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        ## load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

        self.step_counter = 0

    def forward(self, features, kmer_data_1, kmer_data_2):
        '''
        features shape: [batch_size, 6]
        kmer_data shape: [batch_size, 996]
        '''
        h1 = self.embedding_1(kmer_data_1)
        h1 = self.pos_encoding(h1)                    # output shape: (batch_size, 996, 16)
        h2 = self.embedding_1(kmer_data_2)
        h2 = self.pos_encoding(h2)                    # output shape: (batch_size, 996, 16)

        h1 = h1.transpose(1, 2)
        h1 = self.slice_layer_0(h1)                  # output shape: (batch_size, 64, 512)
        h2 = h2.transpose(1, 2)
        h2 = self.slice_layer_0(h2)                  # output shape: (batch_size, 64, 512)

        h1 = self.anchor1(h1)                  # output shape: (batch_size, 256, 128)
        h2 = self.anchor2(h2)                  # output shape: (batch_size, 256, 128)
        h = torch.cat([h1, h2], dim=2)         # output shape: (batch_size, 256, 256)
        h = self.kmer_layer(h)                      # output shape: (batch_size, 32)

        features_18 = self.onehot_layer(features)
        dl_outputs = torch.Tensor().cuda()
        for i, branch in enumerate(self.ensemble_module_list):
            output = branch(features_18[:, self.feature_combination[i]])
            dl_outputs = torch.cat((dl_outputs, output), dim=-1)

        self.step_counter += 1
        if self.step_counter == 87+1:
            print(f"ML model is used. step_counter=={self.step_counter}")
        if self.step_counter > 87:
            cpu_features = features.cpu().detach().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml_outputs = self.gbc_model.predict(cpu_features)
            ml_outputs = torch.from_numpy(ml_outputs).cuda().reshape(-1, 1)
            for i in range(len(self.trees)):
                t = self.trees[i][0].predict(cpu_features)
                t = torch.from_numpy(t).cuda().reshape(-1, 1)
                ml_outputs = torch.cat((ml_outputs, t), dim=1)
        else:
            ml_outputs = torch.zeros(dl_outputs.shape[0], len(self.trees)+1).cuda()
        pred_result = torch.cat([h, dl_outputs, ml_outputs], dim=1).float()

        pred_result = self.out_layer(pred_result)                       # output shape: (batch_size, 1)
        return pred_result, dl_outputs

class DL_Loop_Model_v20230618(nn.Module):
    """
    @description: 先把2anchor model训练好，再合到像0614那样的模型中。
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.anchor1 = nn.ModuleList([
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 128, 128)
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 256, 32)
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 8)
        ])
        self.anchor2 = nn.ModuleList([
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.MaxPool1d(kernel_size=4, stride=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4),
        ])
        self.cat_layer_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 256, 64)
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 16)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.cat_layer_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 16)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.cat_layer_3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.single_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(128+len(self.ensemble_module_list)+len(self.trees)+1, 16),
            nn.Linear(160, 32),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        ## load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

    def forward(self, features, kmer_data_1, kmer_data_2):
        '''
        features shape: [batch_size, 6]
        kmer_data shape: [batch_size, 996]
        '''
        h1 = self.embedding_1(kmer_data_1)
        h1 = self.pos_encoding(h1)                    # output shape: (batch_size, 996, 16)
        h2 = self.embedding_1(kmer_data_2)
        h2 = self.pos_encoding(h2)                    # output shape: (batch_size, 996, 16)

        h1 = h1.transpose(1, 2)
        h1 = self.slice_layer_0(h1)                  # output shape: (batch_size, 64, 512)
        h2 = h2.transpose(1, 2)
        h2 = self.slice_layer_0(h2)                  # output shape: (batch_size, 64, 512)

        h1_list, h2_list = list(), list()
        for i in range(len(self.anchor1)):
            h1 = self.anchor1[i](h1)
            if i % 2 == 1:
                h1_list.append(h1)
        for i in range(len(self.anchor2)):
            h2 = self.anchor2[i](h2)
            if i % 2 == 1:
                h2_list.append(h2)
        h_list = [torch.cat([h1_list[i], h2_list[i]], dim=2) for i in range(len(h1_list))]
        h_list = [self.cat_layer_1(h_list[0]), self.cat_layer_2(h_list[1]), self.cat_layer_3(h_list[2]), self.single_layer(h1_list[2]), self.single_layer(h2_list[2])]
        h = torch.cat(h_list, dim=1)         # output shape: (batch_size, 96)

        dl_outputs = torch.zeros((features.shape[0], 199)).cuda() # //

        h = self.out_layer(h)                       # output shape: (batch_size, 1)
        return h, dl_outputs

class DL_Loop_Model_v20230621(nn.Module):
    """
    @description: 与0613的区别是，这个模型把anchor score和ocr score在外面计算的。
    @thop.profile:
    @model parameter number:
    """
    def __init__(self):
        super().__init__()
        EMBEDDING_DIM = 16
        self.embedding_1 = nn.Embedding(num_embeddings=4**5, embedding_dim=EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(embedding_dim=EMBEDDING_DIM, max_seq_len=1000)

        self.slice_layer_0 = SliceLayer1D(slice_len=512, slice_num=4)
        self.anchor1 = nn.ModuleList([
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 128, 128)
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 256, 32)
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 8)
        ])
        self.anchor2 = nn.ModuleList([
            LargeKernelFireBlock1D_v1(input_channels=64, squeeze_channels=32, e_1_channels=32, e_3_channels=64, e_5_channels=32),
            nn.MaxPool1d(kernel_size=4, stride=4),
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4),
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4),
        ])
        self.cat_layer_1 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=128, squeeze_channels=64, e_1_channels=64, e_3_channels=128, e_5_channels=64),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 256, 64)
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 16)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.cat_layer_2 = nn.Sequential(
            LargeKernelFireBlock1D_v1(input_channels=256, squeeze_channels=128, e_1_channels=128, e_3_channels=256, e_5_channels=128),
            nn.MaxPool1d(kernel_size=4, stride=4), # (batch_size, 512, 16)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.cat_layer_3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.single_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU()
        )

        self.seq_out_layer = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(128+len(self.ensemble_module_list)+len(self.trees)+1, 16),
            nn.Linear(160, 32),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.onehot_layer = OneHotConcatModule()
        FEATURE_INDEX = [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17]]
        self.ensemble_module_list = nn.ModuleList()
        self.feature_combination = []
        self.feature_combination = list(itertools.combinations(FEATURE_INDEX, 3))
        temp_comb = list()
        for comb1 in self.feature_combination:
            temp = list()
            for single in comb1:
                temp += single
            temp_comb.append(temp)
        self.feature_combination = temp_comb
        ## 在C_3_14中共有364个组合，其中至少包含前三个特征之一的组合数量为C_3_14 - C_3_11 = 364 - 165 = 199
        self.feature_combination = [i for i in self.feature_combination if set(i).intersection(set([0,1,2,3,4,5,6]))]
        for f_index in self.feature_combination:
            self.ensemble_module_list.append(append_branch(depth=4, feature_num=len(f_index)))
        print("MODEL ensemble_module_list len: ", len(self.ensemble_module_list))

        self.gbc_model = joblib.load(LOOP_MODEL_PATH+'/model/GradientBoostingClassifier_8cellline_healthcancer.joblib')
        self.trees = self.gbc_model.estimators_

        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1+len(self.ensemble_module_list)+len(self.trees)+1, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        ## load embedding parameters
        saved_dict = torch.load(GLOVE_PATH+f'/model/glove_embedding_dim{EMBEDDING_DIM}.pt')
        embedding_dict = {'weight': saved_dict['focal_embeddings.weight']}
        model_dict = self.embedding_1.state_dict()
        model_dict.update(embedding_dict)
        self.embedding_1.load_state_dict(model_dict)

        self.step_counter = 0
        # self.START_ML_STEP = 117*2
        # self.START_ML_STEP = 599*2
        self.START_ML_STEP = 1609*2

    def forward(self, features, kmer_data_1, kmer_data_2):
        '''
        features shape: [batch_size, 6]
        kmer_data shape: [batch_size, 996]
        '''
        h1 = self.embedding_1(kmer_data_1)
        h1 = self.pos_encoding(h1)                    # output shape: (batch_size, 996, 16)
        h2 = self.embedding_1(kmer_data_2)
        h2 = self.pos_encoding(h2)                    # output shape: (batch_size, 996, 16)

        h1 = h1.transpose(1, 2)
        h1 = self.slice_layer_0(h1)                  # output shape: (batch_size, 64, 512)
        h2 = h2.transpose(1, 2)
        h2 = self.slice_layer_0(h2)                  # output shape: (batch_size, 64, 512)

        h1_list, h2_list = list(), list()
        for i in range(len(self.anchor1)):
            h1 = self.anchor1[i](h1)
            if i % 2 == 1:
                h1_list.append(h1)
        for i in range(len(self.anchor2)):
            h2 = self.anchor2[i](h2)
            if i % 2 == 1:
                h2_list.append(h2)
        h_list = [torch.cat([h1_list[i], h2_list[i]], dim=2) for i in range(len(h1_list))]
        h_list = [self.cat_layer_1(h_list[0]), self.cat_layer_2(h_list[1]), self.cat_layer_3(h_list[2]), self.single_layer(h1_list[2]), self.single_layer(h2_list[2])]
        h = torch.cat(h_list, dim=1)         # output shape: (batch_size, 96)
        h = self.seq_out_layer(h)            # output shape: (batch_size, 1)

        features_18 = self.onehot_layer(features)
        dl_outputs = torch.Tensor().cuda()
        for i, branch in enumerate(self.ensemble_module_list):
            output = branch(features_18[:, self.feature_combination[i]])
            dl_outputs = torch.cat((dl_outputs, output), dim=-1)

        self.step_counter += 1
        if self.step_counter == self.START_ML_STEP+1:
            print(f"ML model is used. step_counter=={self.step_counter}")
        if self.step_counter > self.START_ML_STEP:
            cpu_features = features.cpu().detach().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml_outputs = self.gbc_model.predict(cpu_features)
            ml_outputs = torch.from_numpy(ml_outputs).cuda().reshape(-1, 1)
            for i in range(len(self.trees)):
                t = self.trees[i][0].predict(cpu_features)
                t = torch.from_numpy(t).cuda().reshape(-1, 1)
                ml_outputs = torch.cat((ml_outputs, t), dim=1)
        else:
            ml_outputs = torch.randn(dl_outputs.shape[0], len(self.trees)+1).cuda()
        pred_result = torch.cat([h, dl_outputs, ml_outputs], dim=1).float()

        pred_result = self.out_layer(pred_result)                       # output shape: (batch_size, 1)
        return pred_result, torch.cat([h, dl_outputs], dim=1).float()


if __name__=="__main__":
    DL_Loop_Model_v20230618()

