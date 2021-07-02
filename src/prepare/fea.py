# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import gc
import random
import time
import sys 
import os
from utils import reduce_mem, uAUC, ProNE, HyperParam
import logging
import pickle
from gensim.models import word2vec
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
import networkx as nx
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
import warnings


pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


## 数据预处理（去噪）
origin_data_path = '../../data/wedata/wechat_algo_data1/'   # 原始数据目录
feature_path = "../../data/feature/"     # 特征存放目录

if not os.path.exists(feature_path):
    os.mkdir(feature_path)

train = pd.read_csv(origin_data_path + 'user_action.csv')
print(train.shape)
train.drop_duplicates(['userid', 'feedid'], inplace=True)
print(train.shape)
train['play'] = train['play'] / 1000.0
train['stay'] = train['stay'] / 1000.0
train['play'] = train['play'].apply(lambda x: min(x, 180.0))
train['stay'] = train['stay'].apply(lambda x: min(x, 180.0))

test_a = pd.read_csv(origin_data_path + 'test_a.csv')
test_b = pd.read_csv(origin_data_path + 'test_b.csv')
print(test_a.shape, test_b.shape)

feed_info = pd.read_csv(origin_data_path + 'feed_info.csv')
feed_info['videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x: min(x, 60))

train.to_pickle(origin_data_path + "user_action.pkl")
test_a.to_pickle(origin_data_path + "test_a.pkl")
test_b.to_pickle(origin_data_path + "test_b.pkl")
feed_info.to_pickle(origin_data_path + "feed_info.pkl")



'''
## 1. User侧的GNN特征
ProNE： GNN embedding  
针对 feedid、authorid
## 选取play_times >= 0.5的数据，
训练得到user的embedding向量（兴趣向量）
'''
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

## 读取训练集
train = pd.read_pickle(origin_data_path + 'user_action.pkl')
## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = max_day
print(train.shape, test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

feed_info = pd.read_pickle(origin_data_path + 'feed_info.pkl')[['feedid', 'authorid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
# df['play_times'] = df['play'] / df['videoplayseconds']

# df = df[df['play_times'] >= 0.5].reset_index(drop=True)
# print(df.shape)


### userid-feedid二部图
uid_lbl,qid_lbl = LabelEncoder(), LabelEncoder()
df['new_uid'] = uid_lbl.fit_transform(df['userid'])
df['new_fid'] = qid_lbl.fit_transform(df['feedid'])
df['new_fid'] += df['new_uid'].max() + 1
G = nx.Graph()
G.add_edges_from(df[['new_uid','new_fid']].values)

model = ProNE(G, emb_size=16, n_iter=6, step=12) 

features_matrix = model.fit(model.mat, model.mat)
model.chebyshev_gaussian(model.mat, features_matrix,
                         model.step, model.mu, model.theta)
emb = model.transform() 

emb = emb[emb['nodes'].isin(df['new_uid'])]
emb['nodes'] = uid_lbl.inverse_transform(emb['nodes'])
emb.rename(columns={'nodes' : 'userid'}, inplace=True)

for col in emb.columns[1:]:
    emb[col] = emb[col].astype(np.float16)
    
user_prone_emb = emb[emb.columns]
user_prone_emb.columns = ['userid'] + ['fid_prone_emb{}'.format(i) for i in range(16)]


### userid-author二部图
uid_lbl,qid_lbl = LabelEncoder(), LabelEncoder()
df['new_uid'] = uid_lbl.fit_transform(df['userid'])
df['new_fid'] = qid_lbl.fit_transform(df['authorid'])
df['new_fid'] += df['new_uid'].max() + 1
G = nx.Graph()
G.add_edges_from(df[['new_uid','new_fid']].values)

model = ProNE(G, emb_size=16, n_iter=6, step=12) 

features_matrix = model.fit(model.mat, model.mat)
model.chebyshev_gaussian(model.mat, features_matrix,
                         model.step, model.mu, model.theta)
emb = model.transform() 

emb = emb[emb['nodes'].isin(df['new_uid'])]
emb['nodes'] = uid_lbl.inverse_transform(emb['nodes'])
emb.rename(columns={'nodes' : 'userid'}, inplace=True)

for col in emb.columns[1:]:
    emb[col] = emb[col].astype(np.float16)
    
user_prone_emb2 = emb[emb.columns]
user_prone_emb2.columns = ['userid'] + ['aid_prone_emb{}'.format(i) for i in range(16)]

# merge，合并
print(user_prone_emb.shape, user_prone_emb2.shape)
user_prone_emb = user_prone_emb.merge(user_prone_emb2, how='left', on=['userid'])
print(user_prone_emb.shape)

user_prone_emb['userid'] = user_prone_emb['userid'].astype(np.int32)
user_prone_emb.to_pickle(feature_path + "uid_prone_emb.pkl")


# 2 Feedid侧的特征
## mmu多模态特征
feed_emb = pd.read_csv(origin_data_path + "feed_embeddings.csv")
print(feed_emb.shape)
time.sleep(0.5)
feedid_list, emb_list = [], []
for line in tqdm(feed_emb.values):
    fid, emb = int(line[0]), [float(x) for x in line[1].split()]
    feedid_list.append(fid)
    emb_list.append(emb)

feedid_emb = np.array(emb_list, dtype=np.float32)

# ss = StandardScaler()
# feedid_emb = ss.fit_transform(feedid_emb)
# print(feedid_emb.shape)

# pca = PCA(n_components=emb_size)
# fid_emb = pca.fit_transform(feedid_emb)

emb_size = 16
svd = TruncatedSVD(n_components=emb_size)
fid_emb = svd.fit_transform(feedid_emb)
print(fid_emb.shape)
fid_emb = fid_emb.astype(np.float16)

fid_mmu_emb = pd.concat([feed_emb[['feedid']],
                         pd.DataFrame(fid_emb, columns=['mmu_emb{}'.format(i) for i in range(emb_size)])], 
                        axis=1)

fid_mmu_emb.to_pickle(feature_path + "fid_mmu_emb.pkl")



## word2vec特征

## 读取训练集
train = pd.read_pickle(origin_data_path + 'user_action.pkl')
print(train.shape)
# train.drop_duplicates(subset=['userid', 'feedid'], keep='last', inplace=True)
# print(train.shape)

## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

feed_info = pd.read_pickle(origin_data_path + 'feed_info.pkl')[['feedid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']


# 用户历史七天的 feedid序列
user_fid_list = []
max_day = 16
n_day = 7
for target_day in range(8, max_day + 1):
    left, right = max(target_day - n_day, 1), target_day - 1
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
    print(tmp.shape)
    user_dict = tmp.groupby('userid')['feedid'].agg(list)
    user_fid_list.extend(user_dict.values.tolist())
    
#     tmp = tmp[tmp['play_times'] >= 1.0]
#     print(tmp.shape)
#     user_dict = tmp.groupby('userid')['feedid'].agg(list)
#     user_fid_list.extend(user_dict.values.tolist())
#     print(target_day, len(user_fid_list))
    
## 训练word2vec 
print(len(user_fid_list))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(user_fid_list, min_count=1, window=20, vector_size=16, sg=1, workers=32, epochs=5) 


## 将每个feedid的向量保存为pickle
emb_size = 16
feed_emb = pd.read_csv(origin_data_path + "feed_embeddings.csv")[['feedid']]
w2v_fid_mat = []
for fid in tqdm(feed_emb.feedid.values):
    try:
        emb = model.wv[fid]
    except:
        emb = np.zeros(emb_size)
    w2v_fid_mat.append(emb)
w2v_fid_mat = np.array(w2v_fid_mat, dtype=np.float32)

fid_w2v_emb = pd.concat([feed_emb, pd.DataFrame(w2v_fid_mat, 
                                                columns=['w2v_emb{}'.format(i) for i in range(emb_size)])], 
                        axis=1)

fid_w2v_emb.to_pickle(feature_path + "fid_w2v_emb.pkl")


## tfidf-svd特征

# - keyword、tag、desc文本
feed_info = pd.read_pickle(origin_data_path + 'feed_info.pkl')
feed_info.fillna('0', inplace=True)


manual_kw = feed_info['manual_keyword_list'].progress_apply(lambda x: x.split(';'))
machine_kw = feed_info['machine_keyword_list'].progress_apply(lambda x: x.split(';'))

manual_tag = feed_info['manual_tag_list'].progress_apply(lambda x: x.split(';'))
def func(x):
    if x == '0':
        return ['0']
    return [_.split()[0] for _ in x.split(';') if float(_.split()[1]) >= 0.5]
machine_tag = feed_info['machine_tag_list'].progress_apply(lambda x: func(x))

all_kw = [' '.join(manual_kw[i] + machine_kw[i]) for i in range(len(manual_kw))]
all_tag = [' '.join(manual_tag[i] + machine_tag[i]) for i in range(len(manual_tag))]


## 处理keyword
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=10)
all_kw_mat = tfidf_vectorizer.fit_transform(all_kw)
kw1 = np.array(all_kw_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=8)
all_kw_mat = svd.fit_transform(all_kw_mat)
print(all_kw_mat.shape)


## 处理tag
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=3)
all_tag_mat = tfidf_vectorizer.fit_transform(all_tag)
tag1 = np.array(all_tag_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=8)
all_tag_mat = svd.fit_transform(all_tag_mat)
print(all_tag_mat.shape)


## 处理words
all_words = feed_info['description'] + ' ' + feed_info['ocr'] + ' ' + feed_info['asr']
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=5)
all_words_mat = tfidf_vectorizer.fit_transform(all_words.values.tolist())

svd = TruncatedSVD(n_components=8)
all_words_mat = svd.fit_transform(all_words_mat)
print(all_words_mat.shape)


all_kw_mat = all_kw_mat.astype(np.float16)
all_tag_mat = all_tag_mat.astype(np.float16)
all_words_mat = all_words_mat.astype(np.float16)

fid_kw_tag_word_emb = pd.concat([feed_info[['feedid']], 
                                pd.DataFrame(all_kw_mat, columns=['kw_emb{}'.format(i) for i in range(8)]),
                                pd.DataFrame(all_tag_mat, columns=['tag_emb{}'.format(i) for i in range(8)]), 
                                pd.DataFrame(all_words_mat, columns=['word_emb{}'.format(i) for i in range(8)]), 
                                ], axis=1)

fid_kw_tag_word_emb.to_pickle(feature_path + "fid_kw_tag_word_emb.pkl")


## 简单文本特征
## 相同字数占比, desc, ocr, asr字数
def funct(row):
    desc = row['description_char']
    ocr = row['ocr_char']
    desc, ocr = set(desc.split()), set(ocr.split())
    return len(desc & ocr) / min(len(desc), len(ocr))

feed_info['desc_ocr_same_rate'] = feed_info.apply(lambda row: funct(row), axis=1)
feed_info['desc_len'] = feed_info['description_char'].apply(lambda x: len(x.split()))
feed_info['asr_len'] = feed_info['asr_char'].apply(lambda x: len(x.split()))
feed_info['ocr_len'] = feed_info['ocr_char'].apply(lambda x: len(x.split()))

feed_info['keyword1'] = kw1
feed_info['tag1'] = tag1

def get_tag_top1(x):
    try:
        tmp = sorted([(int(x_.split()[0]), float(x_.split()[1]))  for x_ in x.split(';') if len(x_) > 0], 
                       key=lambda x: x[1], reverse=True)
    except:
        return 0
    return tmp[0][0]
feed_info['tag_m1'] =  feed_info['machine_tag_list'].apply(lambda x: get_tag_top1(x))

feed_info.drop(columns=['description', 'ocr', 'asr', 
                        'manual_keyword_list', 'machine_keyword_list', 
                        'manual_tag_list', 'machine_tag_list',
                        'description_char', 'ocr_char', 'asr_char'], inplace=True)
feed_info['bgm_song_id'] = feed_info['bgm_song_id'].astype(np.int32)
feed_info['bgm_singer_id'] = feed_info['bgm_singer_id'].astype(np.int32)
feed_info.to_pickle(feature_path + "feed_info.pkl")



# 过去七天内的 tag、keyword目标编码
## 读取训练集
train = pd.read_pickle(origin_data_path + 'user_action.pkl')
print(train.shape)

## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

## feed侧信息
feed_info = pd.read_csv(origin_data_path + "feed_info.csv")
feed_info.fillna('0', inplace=True)
print(feed_info.shape)



def get_all_kw(row):
    machine_kw = row['machine_keyword_list'].split(';')
    manual_kw = row['manual_keyword_list'].split(';')
    kw = [int(x) for x in machine_kw + manual_kw if x != '0']
    if len(kw) == 0:
        kw = [0]
    return list(set(kw))
    
def get_all_tag(row):
    tmp = row['machine_tag_list']
    
    if tmp == '0':
        machine_tag = ['0']
    else:
        try:
            machine_tag = [x.split()[0] for x in tmp.split(';') if float(x.split()[1]) >= 0.5]
        except:
            print(tmp)
    manual_tag = row['manual_tag_list'].split(';')
    tag = set([int(x) for x in machine_tag + manual_tag if x != '0'])
    if len(tag) == 0:
        tag = [0]
    return list(set(tag))

time.sleep(0.1)
feed_info['all_keyword'] = feed_info.apply(lambda row: get_all_kw(row), axis=1)
feed_info['all_tag'] = feed_info.apply(lambda row: get_all_tag(row), axis=1)

## 去除低频词的kw, tag
kw_all, tag_all = {}, {}
for line in (feed_info['all_keyword'].values):
    for c in line:
        kw_all[c] = kw_all.get(c, 0) + 1

for line in (feed_info['all_tag'].values):
    for c in line:
        tag_all[c] = tag_all.get(c, 0) + 1   
        
print(len(kw_all), len(tag_all))

kw_all = {k: v for k, v in kw_all.items() if v >= 5}
tag_all = {k: v for k, v in tag_all.items() if v >= 5}
print(len(kw_all), len(tag_all))

kw_all = dict(sorted(kw_all.items(), key=lambda x: x[1], reverse=True))
tag_all = dict(sorted(tag_all.items(), key=lambda x: x[1], reverse=True))

def kw_tmp_func(line):
    res = [x for x in line if x in kw_all]
    if len(res) == 0:
        res = [0]
    return res
feed_info['all_keyword'] = feed_info['all_keyword'].apply(lambda line: kw_tmp_func(line))

def tag_tmp_func(line):
    res = [x for x in line if x in tag_all]
    if len(res) == 0:
        res = [0]
    return res
feed_info['all_tag'] = feed_info['all_tag'].apply(lambda line: tag_tmp_func(line))

df = df.merge(feed_info[['feedid', 'all_keyword', 'all_tag']], on='feedid', how='left')

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
print(df.shape)


n_day = 7
max_day = 15

stat_df = pd.DataFrame()
for target_day in range(8, max_day + 1):
    left, right = max(target_day - n_day, 1), target_day - 1
        
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
    tmp = tmp[['all_keyword', 'all_tag'] + y_list[:4]]
    
    # 关键词, tag
    tmp_kw = []
    tmp_tag = []
    for line in tqdm(tmp.values.tolist()):
        kws, tags, y1, y2, y3, y4 = line[0], line[1], line[2], line[3], line[4], line[5]
        for kw in kws:
            tmp_kw.append([kw, y1, y2, y3, y4])
            
        for tg in tags:
            tmp_tag.append([tg, y1, y2, y3, y4])
    
    tmp_kw = pd.DataFrame(tmp_kw, columns=['keyword'] + y_list[:4])
    tmp_kw = tmp_kw.groupby('keyword', as_index=False).agg('mean')
    kw2y1, kw2y2 = dict(zip(tmp_kw['keyword'], np.round(tmp_kw[y_list[0]], 6))), dict(zip(tmp_kw['keyword'], np.round(tmp_kw[y_list[1]], 6)))
    kw2y3, kw2y4 = dict(zip(tmp_kw['keyword'], np.round(tmp_kw[y_list[2]], 6))), dict(zip(tmp_kw['keyword'], np.round(tmp_kw[y_list[3]], 6)))
    
    
    tmp_tag = pd.DataFrame(tmp_tag, columns=['tag'] + y_list[:4])
    tmp_tag = tmp_tag.groupby('tag', as_index=False).agg('mean')
    tag2y1, tag2y2 = dict(zip(tmp_tag['tag'], np.round(tmp_tag[y_list[0]], 6))), dict(zip(tmp_tag['tag'], np.round(tmp_tag[y_list[1]], 6)))
    tag2y3, tag2y4 = dict(zip(tmp_tag['tag'], np.round(tmp_tag[y_list[2]], 6))), dict(zip(tmp_tag['tag'], np.round(tmp_tag[y_list[3]], 6)))
    print(tmp_kw.shape, tmp_tag.shape)
    
    
    cur_df = df[df['date_'] == target_day].reset_index(drop=True)
    cur_df = cur_df[['userid', 'feedid', 'date_', 'all_keyword', 'all_tag']]
    cur_df['kw_y1_ctr'] = cur_df['all_keyword'].apply(lambda xx: [kw2y1.get(x, 0.0) for x in xx])
    cur_df['kw_y2_ctr'] = cur_df['all_keyword'].apply(lambda xx: [kw2y2.get(x, 0.0) for x in xx])
    cur_df['kw_y3_ctr'] = cur_df['all_keyword'].apply(lambda xx: [kw2y3.get(x, 0.0) for x in xx])
    cur_df['kw_y4_ctr'] = cur_df['all_keyword'].apply(lambda xx: [kw2y4.get(x, 0.0) for x in xx])
    
    cur_df['tag_y1_ctr'] = cur_df['all_tag'].apply(lambda xx: [tag2y1.get(x, 0.0) for x in xx])
    cur_df['tag_y2_ctr'] = cur_df['all_tag'].apply(lambda xx: [tag2y2.get(x, 0.0) for x in xx])
    cur_df['tag_y3_ctr'] = cur_df['all_tag'].apply(lambda xx: [tag2y3.get(x, 0.0) for x in xx])
    cur_df['tag_y4_ctr'] = cur_df['all_tag'].apply(lambda xx: [tag2y4.get(x, 0.0) for x in xx])
    
    y1234 = ['kw_y1_ctr', 'kw_y2_ctr', 'kw_y3_ctr', 'kw_y4_ctr', 
             'tag_y1_ctr', 'tag_y2_ctr', 'tag_y3_ctr', 'tag_y4_ctr']
    for fea in tqdm(y1234):
        cur_df[fea + '_mean'] = cur_df[fea].apply(lambda x: np.mean(x))
        cur_df[fea + '_max'] = cur_df[fea].apply(lambda x: np.max(x))

    cur_df = cur_df[['userid', 'feedid', 'date_'] + [x+'_mean' for x in y1234] + [x+'_max' for x in y1234]]
    stat_df = pd.concat([stat_df, cur_df], axis=0, ignore_index=True)
    

stat_df = reduce_mem(stat_df, [f for f in stat_df.columns if f not in ['userid', 'feedid', 'date_']])
stat_df.to_pickle(feature_path + "kw_tag_ctr_stat.pkl")


## target feedid与用户最近历史行为序列的feedid，注意力加权求和
## 读取feedid的embedding向量
fid_w2v_emb = pd.read_pickle(feature_path + "fid_w2v_emb.pkl")
print(fid_w2v_emb.shape)

fid2wvemb = {}
for line in fid_w2v_emb.values:
    fid = int(line[0])
    emb = line[1:]
    fid2wvemb[fid] = emb
    

## 读取训练集
max_day = 15
n_day = 7

train = pd.read_pickle(origin_data_path + 'user_action.pkl')
print(train.shape)

## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
df = df[['userid', 'feedid', 'date_', 'play']]
print(df.shape)


feed_info = pd.read_pickle(origin_data_path + 'feed_info.pkl')[['feedid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']
print(df.shape)

del train, test
gc.collect()


def calc_weight_sum_vector(uid, target_fid, uid2flist):
    query_vec = fid2wvemb[target_fid]
    query_vec = query_vec.reshape(1, -1)
    try:
        key_vecs = np.array([list(fid2wvemb[fid]) for fid in uid2flist[uid]])
    except:
        return np.zeros(16)
    
    res = query_vec @ key_vecs.T
    res = res.reshape(-1)

    def softmax(x):
        row_max = x.max()
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    att = softmax(res)
    att = att.reshape(-1, 1)

    res = np.sum(att * key_vecs, axis=0)
    return res


fid_din_emb = pd.DataFrame()


for target_day in range(8, max_day + 1):
    left, right = max(target_day - n_day, 1), target_day - 1
    
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))]
    tmp = tmp[tmp['play_times'] >= 0.5].reset_index(drop=True)
    tmp['date_'] = target_day
    
    uid2flist = tmp.groupby('userid')['feedid'].agg(list)
    uid2flist = {k: v[-50:] for k, v in uid2flist.items()}
    
    tmp_df = df[df['date_'] == target_day][['userid', 'feedid', 'date_']]
    tmp_df = tmp_df.reset_index(drop=True)
    att_res = []
    for line in tqdm(tmp_df.values):
        uid, fid = int(line[0]), int(line[1])
        din_vec = calc_weight_sum_vector(uid, fid, uid2flist)
        att_res.append(list(din_vec) + list(fid2wvemb[fid] - din_vec))
    
    att_res = np.array(att_res, dtype=np.float16)
    print(att_res.shape)
    din_df = pd.DataFrame(att_res, columns=['din_emb{}'.format(i) for i in range(32)])
    tmp_din_df = pd.concat([tmp_df, din_df], axis=1)
    
    fid_din_emb = pd.concat([fid_din_emb, tmp_din_df], ignore_index=True)

fid_din_emb.to_pickle(feature_path + "fid_din_emb.pkl")


## target feedid与用户最近历史行为序列的feedid, 之差

def w2v_sent2vec(sentence):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in sentence:
        M.append(fid2wvemb[word])
    if len(M) == 0:
        return ((-1 / np.sqrt(embed_size)) * np.ones(embed_size)).astype(np.float32)
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32)


fid_diff_emb = pd.DataFrame()
for target_day in range(8, max_day + 1):
    left, right = max(target_day - n_day, 1), target_day - 1
    
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
    tmp = tmp[tmp['play_times'] >= 0.5]
    tmp['date_'] = target_day
    
    uid2flist = tmp.groupby('userid')['feedid'].agg(list)
    uid2flist = {k: v[-50:] for k, v in uid2flist.items()}
    
    tmp_df = df[df['date_'] == target_day][['userid', 'feedid', 'date_']]
    tmp_df = tmp_df.reset_index(drop=True)
    diff_res = []
    hist_res = []
    for line in tqdm(tmp_df.values):
        uid, fid = int(line[0]), int(line[1])
        if uid not in uid2flist:
            hist_vec = np.zeros(16)
        else:
            hist_vec = w2v_sent2vec(uid2flist.get(uid))
        cur_vec = fid2wvemb[fid]
        
        hist_res.append(hist_vec)
        diff_res.append(cur_vec - hist_vec)
    
    hist_res = np.array(hist_res, dtype=np.float16)
    diff_res = np.array(diff_res, dtype=np.float16)
    
    hist_res = pd.DataFrame(hist_res, columns=['hist_emb{}'.format(i) for i in range(16)])
    diff_res = pd.DataFrame(diff_res, columns=['diff_emb{}'.format(i) for i in range(16)])
    tmp_df = pd.concat([tmp_df, diff_res, hist_res], axis=1)
    
    fid_diff_emb = pd.concat([fid_diff_emb, tmp_df], ignore_index=True)
    

fid_diff_emb.to_pickle(feature_path + "fid_diff_emb.pkl")


## 用户过去的历史序列
## 读取训练集
max_day = 15
n_day = 7

train = pd.read_pickle(origin_data_path + 'user_action.pkl')
print(train.shape)

## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
df = df[['userid', 'feedid', 'date_', 'play']]
print(df.shape)


feed_info = pd.read_pickle(origin_data_path + 'feed_info.pkl')[['feedid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']
print(df.shape)

del train, test
gc.collect()

user_hist_fids = {}

for target_day in range(8, max_day + 1):
    print("target day {}".format(target_day))
    left, right = max(target_day - n_day, 1), target_day - 1
    
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))]
    tmp = tmp[tmp['play_times'] >= 1.0].reset_index(drop=True)
    tmp['date_'] = target_day
    
    uid2flist = tmp.groupby('userid')['feedid'].agg(list)
    uid2flist = {k: v[-100:] for k, v in uid2flist.items()}
    
    tmp_df = df[df['date_'] == target_day][['userid', 'date_']]
    tmp_df.drop_duplicates(inplace=True)
    tmp_df = tmp_df.reset_index(drop=True)
    print(tmp_df.shape)
    
    for line in (tmp_df.values):
        uid, dt = int(line[0]), int(line[1])
        user_hist_fids[(uid, dt)] = uid2flist.get(uid, [0])
        

pickle.dump(user_hist_fids, open(feature_path + "user_hist_fid_seq.pkl", 'wb'))




# 3 统计特征 和 CTR特征
## 读取训练集
train = pd.read_pickle(origin_data_path + 'user_action.pkl')
print(train.shape)
    
## 读取测试集
test = pd.read_pickle(origin_data_path + 'test_b.pkl')
test['date_'] = 15
print(test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)

# feed侧信息
feed_info = pd.read_pickle(feature_path + "feed_info.pkl")
print(feed_info.shape)

df = df.merge(feed_info, on='feedid', how='left')

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
df['stay_times'] = df['stay'] / df['videoplayseconds']

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15


## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 7
max_day = 15
start_time = time.time()


for stat_cols in ([
    ['userid'], ['feedid'], ['authorid'], ['bgm_song_id'], ['bgm_singer_id'], ['tag1'], ['tag_m1'], ['keyword1'],
    ['userid', 'tag1'], ['userid', 'tag_m1'], ['userid', 'keyword1'], 
    ['userid', 'authorid']]):
    
    f = '_'.join(stat_cols)
    print('======== ' + f + ' =========')
    stat_df = pd.DataFrame()
    for target_day in range(8, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        
        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')  # 观看完成率
        
        # 特征列
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
        
        for x in play_cols[1:]:
            for stat in ['max', 'mean', 'sum']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in y_list[:4]:
#             tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
#             tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

            tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'] = g[y].transform('count')
            tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'] = g[y].transform('sum')
            
            HP = HyperParam(1, 1)
            HP.update_from_data_by_moment(tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'].values, 
                                          tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'].values)
            tmp['{}_{}day_{}_ctr'.format(f, n_day, y)] = (tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count']
                                                          + HP.alpha) / (tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'] + HP.alpha + HP.beta)
        
            feats.extend(['{}_{}day_{}_ctr'.format(f, n_day, y)])
        
        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
    
    stat_df = reduce_mem(stat_df, [f for f in stat_df.columns if f not in stat_cols + ['date_'] + play_cols + y_list])
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()
    print("time costed: {}".format(round(time.time() - start_time, 2)))



# 过去七天内的交叉统计

n_day = 7
max_day = 15
start = time.time()

for stat_cols in ([['userid', 'authorid'],
                    ['userid', 'tag1'], ['userid', 'tag_m1'], ['userid', 'keyword1'],
                    ['userid', 'bgm_song_id'], ['userid', 'bgm_singer_id']]):
    print(stat_cols)
    
    f1, f2 = stat_cols
    stat_df = pd.DataFrame()
    for target_day in range(8, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        
        tmp['{}_{}day_count'.format(f1, n_day)] = tmp.groupby(f1)['date_'].transform('count')
        tmp['{}_{}day_count'.format(f2, n_day)] = tmp.groupby(f2)['date_'].transform('count')
        
        tmp['{}_in_{}_{}day_nunique'.format(f1, f2, n_day)] = tmp.groupby(f2)[f1].transform('nunique')
        tmp['{}_in_{}_{}day_nunique'.format(f2, f1, n_day)] = tmp.groupby(f1)[f2].transform('nunique')
        
        tmp['{}_{}_{}day_count'.format(f1, f2, n_day)] = tmp.groupby([f1, f2])['date_'].transform('count')
        tmp['{}_in_{}_{}day_count_prop'.format(f1, f2, n_day)] = tmp['{}_{}_{}day_count'.format(f1, f2, n_day)] / (tmp['{}_{}day_count'.format(f2, n_day)] + 1)
        tmp['{}_in_{}_{}day_count_prop'.format(f2, f1, n_day)] = tmp['{}_{}_{}day_count'.format(f1, f2, n_day)] / (tmp['{}_{}day_count'.format(f1, n_day)] + 1)
        
       
        # 特征列
        feats = ['{}_in_{}_{}day_nunique'.format(f1, f2, n_day), 
                 '{}_in_{}_{}day_nunique'.format(f2, f1, n_day),
                 '{}_in_{}_{}day_count_prop'.format(f1, f2, n_day), 
                 '{}_in_{}_{}day_count_prop'.format(f2, f1, n_day)]
        
        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        
        del tmp
        gc.collect()
    
    stat_df = reduce_mem(stat_df, [f for f in stat_df.columns if f not in stat_cols + ['date_'] + play_cols + y_list])
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()
    
    print("time costed: {}".format(round(time.time() - start_time)))
    
    

## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

for f in tqdm(['userid', 'feedid', 'authorid', 'tag1', 'keyword1', 'bgm_song_id', 'bgm_singer_id']):
    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([
     ['userid', 'feedid'], ['userid', 'authorid'], ['userid', 'tag1'], ['userid', 'keyword1'], 
     ['userid', 'bgm_song_id'], ['userid', 'bgm_singer_id'],]):
    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

    
for f1, f2 in tqdm([['userid', 'authorid'], ['userid', 'tag1'], ['userid', 'keyword1'],
                    ['userid', 'bgm_song_id'], ['userid', 'bgm_singer_id'],]):
    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_keyword1_mean'] = df.groupby('keyword1')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_tag1_mean'] = df.groupby('tag1')['videoplayseconds'].transform('mean')

df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')



## 保存特征
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
print(df.shape)

df[(df['date_'] >= 8) & (df['date_'] <= 14)].to_pickle(feature_path + "train.pkl")
df[df['date_'] == 15].to_pickle(feature_path + "test.pkl")




