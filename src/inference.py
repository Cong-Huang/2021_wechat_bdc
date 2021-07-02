import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold 
import gc
import random
import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import lightgbm as lgb
from collections import defaultdict
from train.utils import reduce_mem, uAUC
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.deepfm import DNN, FM, combined_dnn_input
from deepctr_torch.layers.interaction import FM, BiInteractionPooling
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer
from deepctr_torch.models import AutoInt, xDeepFM, DeepFM

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import pickle
import os
import warnings
from model.model import MyDeepFM

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', None)



"""For LGB"""
print("LGB 预测")
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
feature_path = "../data/feature/"
test = pd.read_pickle(feature_path + "test4lgb.pkl")
print(test.shape)
lgb_save_path = "../data/model/lgb_save/"
action2fea_cols = pickle.load(open(lgb_save_path + "lgb_model_action2fea_cols.pkl", 'rb'))

for y in y_list[:4]:
    test[y] = 0

n_splits=10
for n_fold in range(1, n_splits+1):
    print("fold {}".format(n_fold))
    for y in y_list[:4]:
        print('=========', y, '=========')
        start_time = time.time()
        cols_new = action2fea_cols[y]
        print(len(cols_new))
        clf = lgb.Booster(model_file=lgb_save_path + "lgb_model_{}_{}_fold.txt".format(y, n_fold))
        test[y] += clf.predict(test[cols_new]) / n_splits
        print('runtime: {}(s)\n'.format(round(time.time() - start_time, 6)))
        

submit_lgb = pd.read_pickle("../data/wedata/wechat_algo_data1/test_b.pkl")[['userid', 'feedid']]
submit_lgb['read_comment'] = np.round(test['read_comment'].values, 8)
submit_lgb['like'] = np.round(test['like'].values, 8)
submit_lgb['click_avatar'] = np.round(test['click_avatar'].values, 8)
submit_lgb['forward'] = np.round(test['forward'].values, 8)



"""For NN"""
print("NN 预测")
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
feature_path = "../data/feature/"
test = pd.read_pickle(feature_path + "test4nn.pkl")
print(test.shape)
nn_save_path = "../data/model/nn_save/"
sparse_features, dense_features, fixlen_feature_columns = pickle.load(open(nn_save_path + "fixlen_feature_columns.pkl", 'rb'))


## 定义模型列
embedding_dim = 32
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
print("Feature nums is {}".format(len(feature_names)))


def predict(model, test_loader, device):
    model.eval()
    pred_ans = []
    with torch.no_grad():
        for x in test_loader:
            y_pred = model(x[0].to(device)).cpu().data.numpy().squeeze().tolist()   # .squeeze()
            pred_ans.extend(y_pred)
    return pred_ans

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
print(device)


submit_nn = pd.read_pickle("../data/wedata/wechat_algo_data1/test_b.pkl")[['userid', 'feedid']]
for y in y_list[:4]:
    submit_nn[y] = 0


n_splits = 10
for n_fold in range(1, n_splits+1):
    start_time = time.time()
    print("fold {}".format(n_fold))
    
    for action in ['read_comment', 'like', 'click_avatar', 'forward']:
        print("\n开始处理 {}".format(action))
        # 定义模型
        model = MyDeepFM(linear_feature_columns=linear_feature_columns, 
                     dnn_feature_columns=dnn_feature_columns,
                     dnn_use_bn=True,  
                     emb_size=embedding_dim,
                     dnn_hidden_units=(1024, 512, 256, 128), l2_reg_linear=1e-5, init_std=0.001, 
                     dnn_dropout=0.5,
                     task='binary', l2_reg_embedding=1e-4, device=device,)
        
        
        
        test_dataset = Data.TensorDataset(torch.FloatTensor(np.concatenate((test[sparse_features],
                                                                    test[dense_features]), axis=-1)),)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=8192, shuffle=False, num_workers=0) 
#         print("test loader nums: {}".format(len(test_loader)))
    
        # 优化器和训练模型
        model_save_path = nn_save_path + "best_deepfm_{}_{}_fold.bin".format(action, n_fold)
        # 加载最优模型
        model.load_state_dict(torch.load(model_save_path))
        
        test_y_perd = predict(model, test_loader, device)
        submit_nn[action] += np.round(test_y_perd, 8) / n_splits
        
        del test_loader
        gc.collect()
    
    print("time costed: {}".format(round(time.time() - start_time, 6)))
    gc.collect()
    

## 融合
cols = submit_lgb.columns.tolist()
submit_ronghe = submit_lgb[cols[:2]]

for col in cols[2:]:
    submit_ronghe[col] = np.round(submit_lgb[col] * 0.45 + submit_nn[col] * 0.55, 8)

## 保存
submit_ronghe.to_csv("../data/submission/result.csv", index=None)







