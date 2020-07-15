import torch
from torch import nn,functional,optim
import random
class LFM(nn.Module):
    def __init__(self,user_id,item_id):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lambd = 0.01
        self._init_data(user_id,item_id)
        self.p = nn.Embedding(len(self.user_ids_set),self.class_count)
        self.q = nn.Embedding(len(self.item_ids_set),self.class_count)
        self.loss_fn = torch.nn.MSELoss(reduce=True)


    def _get_dic(self, user_ids, item_ids):
        items_dict = {}
        for user_id in self.user_ids_set:
            items_dict[user_id] = self._randomSelectNegativeSample(user_id, user_ids, item_ids)
        return items_dict

    def _init_data(self, user_ids, item_ids):
        self.user_ids_set = set(user_ids)
        self.item_ids_set = set(item_ids)
        self.items_dict = self._get_dic(user_ids, item_ids)
    def _randomSelectNegativeSample(self, user_id, user_ids, item_ids):
        items = [x[1] for x in zip(user_ids, item_ids) if x[0] == user_id]
        res = dict()
        for i in items:
            res[i] = 1
        n = 0
        for i in range(len(items) * 3):
            item = item_ids[random.randint(0, len(item_ids) - 1)]
            if item in res:
                continue
            res[item] = 0
            n += 1
            if n > len(items):
                break
        return res
    def forward(self,user_id,item_id):
        self.p = self.p.weight[user_id]
        self.q = self.q.weight[item_id]
        r = torch.sum(torch.mul(self.p,self.q))
        logit = functional.sigmoid(r)
        return logit
    def train(self):
        optimizer = optim.SGD(self.parameters(),lr= self.lr)
        for step in range(self.iter_count):
            for user_id, item_dict in self.items_dict.items():
                item_ids = list(item_dict.keys())
                for item_id in item_ids:
                    optimizer.zero_grad()
                    loss = self.loss_fn(item_dict[item_id],self.forward(user_id,item_id))
                    loss += torch.sum(self.lamda*(self.p**2+(self.q**2)))
                    loss.backward()
                    optimizer.step()
    def predict(self,user_id,items,top_n=10):
        user_item_ids = set(items)
        other_item_ids = self.item_ids_set ^ user_item_ids
        interest_list = [self.forward(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]