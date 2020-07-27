import pandas as pd
from collections import defaultdict
from importlib import import_module
import numpy as np
import torch

from train_eval import train,test

data_dir = './data/'
data_train = pd.read_table(data_dir+'u1.base',names=['users','items','ratings','timestamp'])
data_test = pd.read_table(data_dir+'u1.test',names=['users','items','ratings','timestamp'])

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

model_name = 'svd++'

x = import_module('Model.'+model_name)
config = x.Config()
model = x.Model(config).to(config.device)

data_train['users'] -= 1
data_train['items'] -= 1
data_test['users'] -= 1
data_test['items'] -= 1


def dataloader(config, data_train, data_test):
    R = np.zeros((943, 1682))
    for i in range(943):
        data_i = data_train[data_train['users'] == i]
        items_i = data_i['items']
        ratings = data_i['ratings']
        for item, rate in zip(items_i, ratings):
            #             print(item)
            R[i, item] = rate
    R_matrix = torch.from_numpy(R)

    m_dev = sorted(data_test['users'].unique())
    m_dev_dict = {i: m_dev[i] for i in range(len(m_dev))}
    dev_dict = defaultdict(list)

    dev_R = np.zeros((len(m_dev), 1682))
    for i in range(len(m_dev)):
        data_i = data_test[data_test['users'] == m_dev_dict[i]]
        items_i = data_i['items']
        ratings = data_i['ratings']
        dev_dict[m_dev_dict[i]] += items_i.to_list()
        for item, rate in zip(items_i, ratings):
            dev_R[i, item] = rate

    data_test_dict = defaultdict(list)
    for user in data_test['users'].unique():
        items = data_test[data_test['users'] == user]['items'].values.tolist()
        data_test_dict[user] += items

    dev_R = torch.from_numpy(dev_R)

    R_matrix = R_matrix.float().to(config.device)
    #     m_dev = m_dev.to(config.device)
    dev_R = dev_R.float().to(config.device)
    return R_matrix, m_dev, dev_R, dev_dict

R_matrix,dev_index,dev_R,dev_dict = dataloader(config,data_train,data_test)

train(config,model,R_matrix,dev_R,dev_index)
test(config,model,dev_dict,10)

