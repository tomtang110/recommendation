
import pandas as pd
from collections import defaultdict
from importlib import import_module
import numpy as np

from User_based_and_item_based.metrics import cal_precision_at_k


data_dir = '../data/ml-100k/'
data_train = pd.read_csv(data_dir+'train_data.csv')
data_test = pd.read_csv(data_dir+'test_data.csv')

model_name = 'item_cf'
model = import_module('Model.'+model_name)

user_nb = data_train['user'].nunique()
item_nb = data_train['items'].nunique()


data_test_dict = defaultdict(list)
for user in data_test['user'].unique():
    items = data_test[data_test['user'] == user]['items'].values.tolist()
    data_test_dict[user] += items

preci = 0.0
count = 0
if model_name in ['item_cf','User_cvf']:
    data_train['user'] -= 1
    data_train['items'] -= 1
    data_test['user'] -= 1
    data_test['items'] -= 1
    users_train, items_train = data_train['user'].values, data_train['items'].values
    Model = model.Model(users_train,items_train)
    for user_ in data_test_dict.keys():
        result = Model.recommend(user_, 10, 10)
        pred = [k[0] for k in result]
        metric_prec = cal_precision_at_k(10, pred, data_test_dict[user_])
        preci += metric_prec
        count += 1
elif model_name in ['DMF']:
    data_label = np.zeros((user_nb,item_nb))
    for i in range(len(data_train)):
        data_label[data_train['user'][i],data_train['items'][i]] = data_train['ratings'][i]


# 什么

print('precision: ',preci/count)






