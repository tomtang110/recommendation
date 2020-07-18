
import pandas as pd
from collections import defaultdict
from importlib import import_module
import numpy as np
from sklearn import metrics

def cal_precision_at_k(k, rankedlist, test_matrix):
    test_set = set(test_matrix)
    rank_set = set(rankedlist)
    hit = len(test_set & rank_set)
    return float(hit / k)
def cal_Recall_at_k_for_each_user(k, rankedlist, test_matrix):
    test_set = set(test_matrix)
    rank_set = set(rankedlist)
    hit = len(test_set & rank_set)
    return float(hit / len(test_set))


def cal_ndcg_at_k_for_each_user(k, rankedlist, testlist):
    idcg_k = 0
    dcg_k = 0
    if len(testlist) < k: k = len(testlist)
    for i in range(k):
        idcg_k += 1 / np.log(i + 2, 2)

    s = set(testlist)
    hits = [idx for idx, val in enumerate(rankedlist) if val in s]
    count = len(hits)
    for i in range(count):
        dcg_k += 1 / np.log(hits[i] + 2, 2)

    return float(dcg_k / idcg_k)


data_dir = './data/ml-100k/'
data_train = pd.read_csv(data_dir+'train_data.csv')
data_test = pd.read_csv(data_dir+'test_data.csv')

model_name = 'User_cvf'
model = import_module('codes.'+model_name)

data_train['user'] -= 1
data_train['items'] -= 1
data_test['user'] -= 1
data_test['items'] -= 1
users_train, items_train = data_train['user'].values,data_train['items'].values

data_test_dict = defaultdict(list)
for user in data_test['user'].unique():
    items = data_test[data_test['user'] == user]['items'].values.tolist()
    data_test_dict[user] += items


Model = model.Model(users_train,items_train)

preci = 0.0
count = 0
for user_ in data_test_dict.keys():
    result = Model.recommend(user_,10,10)
    pred = [k[0] for k in result]
    metric_prec = cal_precision_at_k(10,pred,data_test_dict[user_])
    preci += metric_prec
    count += 1

print('precision: ',preci/count)






