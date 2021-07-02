# -*- coding: utf-8 -*-

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
import os
import pickle
from utils import reduce_mem, uAUC

pd.set_option('display.max_columns', None)


feature_path = "../../data/feature/"
train = pd.read_pickle(feature_path + "train.pkl")
print(train.shape)
train.drop_duplicates(['userid', 'feedid'], inplace=True)
test = pd.read_pickle(feature_path + "test.pkl")
print(train.shape, test.shape)


df = pd.concat([train, test], ignore_index=True)
print(df.shape)


## 关联embedding信息
fid_kw_tag_word_emb = pd.read_pickle(feature_path + "fid_kw_tag_word_emb.pkl")
fid_mmu_emb = pd.read_pickle(feature_path + "fid_mmu_emb.pkl")
fid_w2v_emb = pd.read_pickle(feature_path + "fid_w2v_emb.pkl")

uid_prone_emb = pd.read_pickle(feature_path + "uid_prone_emb.pkl")

fid_din_emb = pd.read_pickle(feature_path + "fid_din_emb.pkl")
fid_diff_emb = pd.read_pickle(feature_path + "fid_diff_emb.pkl")
kw_tag_ctr_stat = pd.read_pickle(feature_path + "kw_tag_ctr_stat.pkl")

print(fid_kw_tag_word_emb.shape, fid_mmu_emb.shape, fid_w2v_emb.shape, 
      uid_prone_emb.shape,
      fid_din_emb.shape, fid_diff_emb.shape, kw_tag_ctr_stat.shape)


for tmp in tqdm([fid_kw_tag_word_emb, fid_mmu_emb, fid_w2v_emb]):
    df = df.merge(tmp, how='left', on=['feedid'])
    
df = df.merge(uid_prone_emb, how='left', on=['userid'])

for tmp in tqdm([fid_din_emb, kw_tag_ctr_stat, fid_diff_emb]):
    df = df.merge(tmp, how='left', on=['userid', 'feedid', 'date_'])
print(df.shape)


## 切分训练、验证、测试
cate_cols = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']


for col in tqdm(cate_cols):
    lbl = LabelEncoder()
    df[col] = lbl.fit_transform(df[col])
    
# df = reduce_mem(df, [col for col in df.columns.tolist() if col not in cate_cols + play_cols + y_list])
# print(df.shape)

train = df[df['date_'] <= 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
print(train.shape, test.shape)

test.to_pickle(feature_path + "test4lgb.pkl")


## lgb训练模型所需要的特征列
cols = [f for f in test.columns if (f not in ['date_'] + play_cols + y_list)]
print(len(cols))

del df
gc.collect()


# 多折交叉验证训练
for y in y_list:
    test[y] = 0

n_splits = 10
fold = KFold(n_splits=n_splits, shuffle=True, random_state=2021)
kf_way = fold.split(train)

lgb_save_path = "../../data/model/lgb_save/"
if not os.path.exists(lgb_save_path):
    os.mkdir(lgb_save_path)


print("开始提取每个action的最优特征(筛选特征)")
action2fea_cols = {}
for y in y_list[:4]:
    print('=========', y, '=========')
    clf = LGBMClassifier(
            learning_rate=0.1,
            n_estimators=200,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None',
            n_jobs=32
    )
        
    clf.fit(
            train[train['date_'] != 14][cols], train[train['date_'] != 14][y],
            eval_set=[(train[train['date_'] == 14][cols], train[train['date_'] == 14][y])],
            eval_metric='auc',
            early_stopping_rounds=50,
            verbose=100,
    )
    
    fea_imp = pd.DataFrame({'fea': cols, 'imp': clf.feature_importances_})
    fea_imp.sort_values('imp', inplace=True, ascending=False)
    cols_new = fea_imp['fea'].values.tolist()[:-60]
    action2fea_cols[y] = cols_new
    
pickle.dump(action2fea_cols, open(lgb_save_path + "lgb_model_action2fea_cols.pkl", 'wb'))


print("开始多折交叉训练")
for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
    print("fold {}".format(n_fold))
    
    trn_x = train.iloc[train_idx].reset_index(drop=True)
    val_x = train.iloc[valid_idx].reset_index(drop=True)
    print(trn_x.shape, val_x.shape)
    
    uauc_list = []
    
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        cols_new = action2fea_cols[y]
        print(len(cols_new))
        
        clf = LGBMClassifier(
            learning_rate=0.015,
            n_estimators=10000,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=2021,
            metric='None',
            n_jobs=32,
            min_child_samples=100,
        )

        clf.fit(
            trn_x[cols_new], trn_x[y],
            eval_set=[(val_x[cols_new], val_x[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=500
        )
        
        val_x[y + '_score'] = clf.predict_proba(val_x[cols_new])[:, 1]
        val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'].values)
        uauc_list.append(val_uauc)
        print('{} uauc: {}'.format(y, val_uauc))
        test[y] += clf.predict_proba(test[cols_new])[:, 1] / n_splits
        print('runtime: {}\n'.format(time.time() - t))
        clf.booster_.save_model(lgb_save_path + "lgb_model_{}_{}_fold.txt".format(y, n_fold))
        

    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list, round(weighted_uauc, 6))
    print()
    
    del trn_x, val_x
    gc.collect()
    









