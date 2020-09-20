import pandas as pd
import numpy as np
import torch
from SVD.train_eval import train
from SVD.svd import Model,Config

data_dir = '../data/'
data_train = pd.read_table(data_dir+'u1.base',names=['users','items','ratings','timestamp'])
data_test = pd.read_table(data_dir+'u1.test',names=['users','items','ratings','timestamp'])

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

# 去掉时间戳
data_train = data_train.drop(['timestamp'],axis=1)
data_test = data_test.drop(['timestamp'],axis=1)
print("train shape:",data_train.shape)
print("test shape:",data_test.shape)

#userNo的最大值
userNo=max(data_train['users'].max(),data_test['users'].max())+1
print("userNo:",userNo)

#movieNo的最大值
itemNo=max(data_train['items'].max(),data_test['items'].max())+1
print("itemNo:",itemNo)

rating_train=torch.zeros((itemNo,userNo))
rating_test=torch.zeros((itemNo,userNo))

for index,row in data_train.iterrows():
    #train数据集进行遍历
    rating_train[int(row['items'])][int(row['users'])]=row['ratings']

print(rating_train[0:3][1:10])
for index,row in data_test.iterrows():
    rating_test[int(row['items'])][int(row['users'])] = row['ratings']

def normalizeRating(rating_train):
    m,n=rating_train.shape
    # 每部电影的平均得分
    rating_mean=torch.zeros((m,1))
    #所有电影的评分
    all_mean=0
    for i in range(m):
        #每部电影的评分
        idx=(rating_train[i,:]!=0)
        rating_mean[i]=torch.mean(rating_train[i,idx])
    tmp=rating_mean.numpy()
    tmp=np.nan_to_num(tmp)        #对值为NaN进行处理，改成数值0
    rating_mean=torch.tensor(tmp)
    no_zero_rating=np.nonzero(tmp)                #numpyy提取非0元素的位置
    # print("no_zero_rating:",no_zero_rating)
    no_zero_num=np.shape(no_zero_rating)[1]   #非零元素的个数
    print("no_zero_num:",no_zero_num)
    all_mean=torch.sum(rating_mean)/no_zero_num
    return rating_mean,all_mean

rating_mean,all_mean=normalizeRating(rating_train)
print("all mean:",all_mean)
config = Config()
model = Model(config).to(config.device)
train(config,model,rating_train,rating_test)