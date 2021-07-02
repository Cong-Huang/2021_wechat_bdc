# **2021中国高校计算机大赛-微信大数据挑战赛 初赛第28名**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。

大赛官方网站：https://algo.weixin.qq.com/

队伍ID：f8f99967d752487196bbfa3e5ee8646a

队伍名: HL

## **1. 环境配置**

- deepctr-torch==0.2.7
- gensim==4.0.1
- lightgbm==3.2.1
- networkx==2.5.1
- numba==0.53.1
- numpy==1.19.2
- pandas==1.1.0
- scikit-learn==0.24.2
- scipy==1.5.3
- sklearn==0.24.2
- torch==1.6.0
- tqdm==4.59.0
- transformers==4.4.2

## **2. 运行配置**

- GPU： 8G显存
- 最小内存要求
    - 特征/样本生成：48G
    - 模型训练及评估：48G
    
    
## **3. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──fea.py   
|       ├──utils.py   
│   ├── model, codes for model architecture
|       ├──model.py  
|   ├── train, codes for training
|       ├──lgb.py  
|       ├──nn.py  
|       ├──utils.py  
|   ├── inference.py, main function for inference on test dataset
├── data
│   ├── wedata, dataset of the competition
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. tensorflow checkpoints)
│   ├── feature, all features
```

## **4. 运行流程**
- 安装环境：sh init.sh
- 将初赛数据集解压后放到data/wedata目录下，得到data/wedata/
- 数据准备和模型训练：sh train.sh
- 预测并生成结果文件：sh inference.sh

## **5. 模型及特征**
  模型：DeepFM, LGB

  特征：

（1）feed侧的特征
- 512维多模态特征降维到16维特征（SVD降维）
- word2vec 遍历用户点击序列， 训练16维特征
- keyword的TFIDF->SVD 降维 8维
- tag 的TFIDF-SVD降维到8维
- keyword1：挑选tfidf值第一高的
- tag1: 挑选tfidf值第一高的
- tag_m1: 挑选tag，机器打分最高的标签
- desc_ocr_sim:  相同字的个数占描述字数的比例（劣势）


（2）user侧的特征
- user-feedid二部图的ProNE图特征 16维
- User-authorid  二部图的ProNE图特征 16维

（3）点击率特征 （target encoding）
- ['userid'], ['feedid'], ['authorid'], [bgm_song], [‘bgm_singer’]
    ['machine_tag1'], ['keyword1'],['userid', 'machine_tag1'],  ['userid', 'authorid']  等等  
    （过去七天内历史特征，防止穿越）,为了更好的计算点击率，降低不确定性，加了贝叶斯平滑。

（4）统计特征
- 过去七天内：
    ['is_finish', 'play_times', 'play', 'stay'] 的统计特征  （min, max, std, mean, sum）
- 全局统计：count、nunique、mean等

（5) embedding特征（都是在w2v的embedding基础上做的）
- DIN特征(16维， 直接点积计算得到分数，再softmax得到归一化权重，最后加权求和，得到用户历史兴趣向量）
- 当前feedid的embedding  - DIN特征。 （交叉，交互）
- Mean特征（base，历史feedid的平均）  ，类似句向量操作
- 当前feedid的embedding  - Mean特征。 （交叉，交互）

## **6. 模型结果**
线上结果
|得分|	查看评论|	点赞|	点击头像|	转发|
|:---|:---|:---|:---|:---|
|0.678666|	0.64929|	0.656366|	0.751049|	0.718303|


## **7. 相关文献**
* Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017).
* Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." Advances in neural information processing systems 30 (2017): 3146-3154.
