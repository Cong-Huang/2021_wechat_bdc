# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from collections import defaultdict
import gc, time
from utils import reduce_mem, uAUC
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
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', None)


feature_path = "../../data/feature/"
train = pd.read_pickle(feature_path + "train.pkl").reset_index(drop=True)
test = pd.read_pickle(feature_path + "test.pkl").reset_index(drop=True)

print(train.shape, test.shape)

df = pd.concat([train, test], ignore_index=True)
print(df.shape)



## 特征列的定义
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
tmp_cols = [f for f in train.columns if f not in ['date_'] + play_cols + y_list]
print("origin feas nums: {}".format(len(tmp_cols)))

fea_cols = []
for col in tmp_cols:
    if round(train[col].isnull().sum() / train.shape[0], 4) <= 0.5:
        fea_cols.append(col)

sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 
                   'bgm_singer_id', 'keyword1', 'tag1', 'tag_m1']
dense_features = [x for x in fea_cols if (x not in sparse_features)]
print("ID feas nums: {}, dense feas nums: {}".format(len(sparse_features), len(dense_features)))

del_features = [x for x in train.columns if x not in y_list+sparse_features+dense_features+['date_']]
print("delete feas nums: {}".format(len(del_features)))

df.drop(columns=del_features, inplace=True)
print(df.shape)


time.sleep(0.5)
## 归一化
for col in tqdm(dense_features):
    x = df[col].astype(np.float32)
    x = np.log(x + 1.0)
    minmaxscalar = MinMaxScaler()
    x = minmaxscalar.fit_transform(x.values.reshape(-1, 1))
    df[col] = x.reshape(-1).astype(np.float16)
    
df[dense_features] = df[dense_features].fillna(0)


## 标准化
# for col in tqdm(dense_features):
#     x = df[col].astype(np.float32)
#     if x.min() >= 0:
#         x = np.log(x + 1.0)
#     mean = x.mean()
#     std = x.std()
#     x = (x - mean) / (std + 1e-8)
#     df[col] = x

# df[dense_features] = df[dense_features].fillna(0)


## merge其他的特征
fid_kw_tag_word_emb = pd.read_pickle(feature_path + "fid_kw_tag_word_emb.pkl")
fid_mmu_emb = pd.read_pickle(feature_path + "fid_mmu_emb.pkl")
fid_w2v_emb = pd.read_pickle(feature_path + "fid_w2v_emb.pkl")
fid_din_emb = pd.read_pickle(feature_path + "fid_din_emb.pkl")
fid_diff_emb = pd.read_pickle(feature_path + "fid_diff_emb.pkl")
uid_prone_emb = pd.read_pickle(feature_path + "uid_prone_emb.pkl")
kw_tag_ctr_stat = pd.read_pickle(feature_path + "kw_tag_ctr_stat.pkl")
print(fid_kw_tag_word_emb.shape, fid_mmu_emb.shape, uid_prone_emb.shape,
      fid_w2v_emb.shape, fid_din_emb.shape,
      kw_tag_ctr_stat.shape)

time.sleep(0.5)
for tmp in tqdm([fid_kw_tag_word_emb, fid_mmu_emb, fid_w2v_emb]):
    df = df.merge(tmp, how='left', on=['feedid'])
    
df = df.merge(uid_prone_emb, how='left', on=['userid'])

for tmp in tqdm([fid_din_emb, fid_diff_emb, kw_tag_ctr_stat]):
    df = df.merge(tmp, how='left', on=['userid', 'feedid', 'date_'])

print(df.shape)

for col in (sparse_features):
    df[col] = df[col].map(dict(zip(df[col].unique(), range(df[col].nunique())))).astype(np.int32)


## 数值特征归一化
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

tmp_cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]
print(len(tmp_cols))

sparse_features = ['userid', 'feedid', 'device', 'authorid', 
                   'bgm_song_id', 'bgm_singer_id', 'keyword1', 'tag1', 'tag_m1']

# 对类别变量和数值特征进行处理
dense_features = [x for x in tmp_cols if x not in sparse_features]
print(len(sparse_features), len(dense_features))

df[dense_features] = df[dense_features].fillna(0)


# df = reduce_mem(df, sparse_features + dense_features)
# print(df.shape)

train = df[df['date_'] <= 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
print(train.shape, test.shape)

test.to_pickle(feature_path + "test4nn.pkl")

del df
gc.collect()


class MyDeepFM(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, 
                 use_fm=True, use_din=False,
                 emb_size=32,
                 dnn_hidden_units=(256, 128), 
                 l2_reg_linear=0.0001, l2_reg_embedding=0.01, l2_reg_dnn=0.0, init_std=0.001, seed=1024,
                 dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None):
        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)
            
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        
        
        ## 暂时没用（For 复赛）
        self.use_din = use_din
        if use_din:
            self.feedid_emb_din = self.embedding_dict.feedid
            self.LSTM_din = nn.LSTM(input_size=emb_size, hidden_size=emb_size, num_layers=1,
                                    batch_first=True, bidirectional=False)
            self.attention = AttentionSequencePoolingLayer(att_hidden_units=(64, 64),
                                                       embedding_dim=emb_size,
                                                       att_activation='Dice',
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=False)
            self.din_linear = nn.Linear(emb_size, 1, bias=False)
        
        self.to(device) 
    

    def forward(self, X, fids=None, fids_length=None):
        """fids: [bs, seqlen]
            fids_length [bs, 1]
        """
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)   # [bs, 1]
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)   # [bs, n, emb_dim]
            logit += self.fm(fm_input)   # [bs, 1]

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)   # [bs, dnn_hidden_units[-1]]
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit   # [bs, 1]
            
        if self.use_din:
            fid_emb_query = self.feedid_emb_din(X[:, self.feature_index['feedid'][0]:self.feature_index['feedid'][1]].long())
            fid_emb_key = self.feedid_emb_din(fids)    # [bs, sl, emb_size]
            fid_emb_key, _ = self.LSTM_din(fid_emb_key)   # [bs, sl, emb_size]
            fid_din = self.attention(fid_emb_query, fid_emb_key, fids_length)   #[bs, 1, emb_size]
            din_logit = self.din_linear(fid_din.squeeze(1))
            logit += din_logit

        y_pred = self.out(logit)   # sigmoid 转化为概率
        return y_pred
    
    
def get_data_loader(train, valid, test, sparse_features, dense_features, target, batch_size=4096):
    train_dataset = Data.TensorDataset(torch.FloatTensor(np.concatenate((train[sparse_features],
                                                                        train[dense_features]), axis=-1)),
                                        torch.FloatTensor(train[target].values))
    
    valid_dataset = Data.TensorDataset(torch.FloatTensor(np.concatenate((valid[sparse_features],
                                                                        valid[dense_features]), axis=-1)),
                                        torch.FloatTensor(valid[target].values))
    
    test_dataset = Data.TensorDataset(torch.FloatTensor(np.concatenate((test[sparse_features],
                                                                        test[dense_features]), axis=-1)),)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)    
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    return train_loader, valid_loader, test_loader


# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def predict(model, test_loader, device):
    model.eval()
    pred_ans = []
    with torch.no_grad():
        for x in test_loader:
            y_pred = model(x[0].to(device)).cpu().data.numpy().squeeze().tolist()   # .squeeze()
            pred_ans.extend(y_pred)
    return pred_ans


def evaluate(model, valid_loader, valid_y, valid_users, device):
    pred_ans = predict(model, valid_loader, device)
    eval_result = uAUC(valid_y, pred_ans, valid_users)
    return eval_result


def train_model(action, model, train_loader, valid_loader, valid,
                optimizer, epochs, device, save_path):
    train_bs = len(train_loader)
    best_score = 0.0
    
    patience = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss_sum = 0
        time.sleep(1.0)
        for idx, (x_train, y_train) in tqdm(enumerate(train_loader)):
            x = x_train.to(device)
            y = y_train.to(device)
            y_pred = model(x).squeeze()

            optimizer.zero_grad()
            loss = F.binary_cross_entropy(y_pred, y.squeeze(), reduction='mean')
            reg_loss = model.get_regularization_loss()

            total_loss = loss + reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss_sum += total_loss.item()
            
            if (idx + 1) == train_bs:
                LR = optimizer.state_dict()['param_groups'][0]['lr']
                print("Epoch {:03d} | Step {:04d} / {} | Loss {:.4f} | LR {:.5f} | Time {:.4f}".format(
                     epoch+1, idx+1, train_bs, total_loss_sum/(idx+1), LR,
                     time.time() - start_time))
        
        valid_pred = predict(model, valid_loader, device)
        valid_true = valid[action].values
        valid_users = valid['userid'].values
        score = uAUC(valid_true, valid_pred, valid_users) 
        print("Epoch:{} 结束，验证集AUC = {}".format(epoch + 1, score))
        
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
        print("Valid cur uAUC = {}, Valid best uAUC = {}, Time {:.2f}".format(score, best_score, 
                                                                              time.time() - start_time))
        
        if patience >= 3:
            print("Early Stopped! ")
            break
            


from sklearn.model_selection import KFold 
n_splits = 10
fold = KFold(n_splits=n_splits, shuffle=True, random_state=2021)
kf_way = fold.split(train)


nn_save_path = "../../data/model/nn_save/"
if not os.path.exists(nn_save_path):
    os.mkdir(nn_save_path)



submit = pd.read_pickle(feature_path + "test.pkl")[['userid', 'feedid']]
for y in y_list[:4]:
    submit[y] = 0

## 定义模型列
embedding_dim = 32
# count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, max(train[feat].max(), test[feat].max())+1, 
                                         embedding_dim=embedding_dim)
                                   for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
# 所有特征列， dnn和linear都一样
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)   # all-特征名字
print("Feature nums is {}".format(len(feature_names)))
pickle.dump([sparse_features, dense_features, fixlen_feature_columns], 
            open(nn_save_path + "fixlen_feature_columns.pkl", 'wb'))


device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
    
for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
    start_time = time.time()
    print("fold {}".format(n_fold))
    trn_x = train.iloc[train_idx].reset_index(drop=True)
    val_x = train.iloc[valid_idx].reset_index(drop=True)
    print(trn_x.shape, val_x.shape)

    ## 开始训练模型
    valid_uauc = []
    
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
        
        print(get_parameter_number(model))
    
        # get 数据加载器
        train_loader, valid_loader, test_loader = get_data_loader(trn_x, val_x, test,
                                                              sparse_features, dense_features,
                                                              action, batch_size=4096)
        print(len(train_loader), len(valid_loader), len(test_loader)) 
    
        # 优化器和训练模型
        optimizer = optim.Adagrad(model.parameters(), lr=0.01) 
        num_epochs = 20
        
        model_save_path = nn_save_path + "best_deepfm_{}_{}_fold.bin".format(action, n_fold)
        
        train_model(action, model, train_loader, valid_loader, val_x,
                    optimizer, epochs=num_epochs, device=device, save_path=model_save_path)
        
        # 加载最优模型
        model.load_state_dict(torch.load(model_save_path))
        
        val_score = evaluate(model, valid_loader, val_x[action].values, 
                            val_x['userid'].values, device)
        print("Valid best score is {}".format(val_score))
        
        test_y_perd = predict(model, test_loader, device)
        submit[action] += np.round(test_y_perd, 8) / n_splits
        valid_uauc.append(val_score)
        
        del train_loader, valid_loader, test_loader
        gc.collect()
    
    print(valid_uauc)
    print((valid_uauc[0]*4 + valid_uauc[1]*3 + valid_uauc[2]*2 + valid_uauc[3]*1) / 10)
    
    print("time costed: {}".format(round(time.time() - start_time, 6)))
    
    del trn_x, val_x
    gc.collect()
