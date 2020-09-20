from torch.nn import functional as F
import torch
import time
from sklearn import metrics
from DMF.metrics import cal_precision_at_k,cal_Recall_at_k_for_each_user,cal_ndcg_at_k_for_each_user
import numpy as np
def train(config,model,train_iter=None,dev_iter=None):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = True
    for ep in range(config.epoch):
#         print('Epoch [{}/{}]'.format(ep+1,config.epoch))
        output = model(train_iter[0],train_iter[1])
        model.zero_grad()
        rate = train_iter[2] / 5.0
        loss = rate * torch.log(output) + (1-rate) * torch.log(1-output)
        loss.backward()
        optimizer.step()

        if total_batch % 50 == 0:
            dev_loss = evaluate(model,dev_iter)
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(),config.save_path+'DMF.ckpt')
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''
            print('Iter:{0:>6}, Train Loss:{1:>5.2}, Val Loss:{2:>6.2}, {3:>4.3}'.format(total_batch,loss,dev_loss,improve))
        total_batch += 1
        if total_batch - last_improve > config.require_improvement:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break





def evaluate(model,dev_iter):
    model.evaluate()
    output = model(dev_iter[0], dev_iter[1])
    rate = dev_iter[2] / 5.0
    loss = rate * torch.log(output) + (1 - rate) * torch.log(1 - output)
    model.train()
    return loss

def trans_test_data(config,user_id):
    users = np.array([user_id for i in range(config.n)])
    items = np.array([item for item in range(config.n)])
    users = torch.from_numpy(users)
    items = torch.from_numpy(items)
    users = users.float().to(config.device)
    items = items.float().to(config.device)
    return users,items
def test(config,model,test_index,K):

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    prec = 0.0
    recal = 0.0
    ndgc = 0.0
    for user_ in test_index.keys():
        users,items = trans_test_data(config,user_)
        result = model(users,items)
        result = result.numpy().argsort()[::-1][:K]
        prec += cal_precision_at_k(K,result,test_index[user_])
        recal += cal_Recall_at_k_for_each_user(K,result,test_index[user_])
        ndgc += cal_ndcg_at_k_for_each_user(K,result,test_index[user_])

    print('Precision: {}, Recall: {} NDGC:{}'.format(prec/len(test_index),recal/len(test_index),ndgc/len(test_index)))