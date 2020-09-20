import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class Config(object):
    def __init__(self):
        self.model_name = 'DMF'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m = 943
        self.n = 1682
        self.h_dim = 256
        self.layer = 2
        self.epoch = 10
        self.lr= 1e-3
        self.save_path = './Model_save/'
        self.require_improvement = 100

class Model(nn.Module):

    def __init__(self,embedding,config):
        super(Model,self).__init__()

        self.m = config.m
        self.n = config.n
        self.embedding_user = nn.Embedding.from_pretrained(embedding)
        self.embedding_item = nn.Embedding.from_pretrained(embedding.T)

        self.h_dim = config.h_dim
        self.layers = config.layer
        self.lr = config.lr
        self.epoch = config.epoch

        Linear_users = [nn.Linear(self.n,self.h_dim)]
        for i in range(self.layers):
            Linear_users.append(nn.Linear(self.h_dim,self.h_dim))
            Linear_users.append(nn.ReLU())
        self.Linear_user = nn.Sequential(Linear_users)
        Linear_items = [nn.Linear(self.m,self.h_dim)]

        for i in range(self.layers):
            Linear_items.append(nn.Linear(self.h_dim, self.h_dim))
            Linear_items.append(nn.ReLU())
        self.Linear_item = nn.Sequential(Linear_items)
    def forward(self,user,item):
        users = self.embedding_user(user)
        items = self.embedding_item(item)

        users = self.Linear_user(users)
        items = self.Linear_item(items)

        out = F.cosine_similarity(users,items)
        out = out.where(out<1e-6,torch.tensor(1e-6),out)
        return out
    def train_batch(self,user,item,rate):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        rate = rate / 5.0
        for _ in range(self.epoch):
            optimizer.zero_grad()
            out = self.forward(user,item)

            loss = rate * torch.log(out) + (1-rate) * torch.log(1-out)
            loss.backward()
            optimizer.step()





