# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.deepfm import DNN, FM, combined_dnn_input
from deepctr_torch.layers.interaction import FM, BiInteractionPooling
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from deepctr_torch.layers import DNN, concat_fun, InteractingLayer
from deepctr_torch.models import AutoInt, xDeepFM, DeepFM



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
    