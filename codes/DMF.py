import torch
from torch import nn
from torch import functional as F
from torch import optim

class Model(nn.Module):

    def __init__(self,embedding,m,n,hidden_dim,layer,alpha,lambda_1,beta1,epoch):
        self.m = m
        self.n = n
        embedding = torch.from_numpy(embedding)
        self.embedding_user = nn.Embedding.from_pretrained(embedding)
        self.embedding_item = nn.Embedding.from_pretrained(embedding.T)

        self.h_dim = hidden_dim
        self.layers = layer
        self.lr = alpha
        self.epoch = epoch

        Linear_users = [nn.Linear(n,self.h_dim)]
        for i in range(self.layers):
            Linear_users.append(nn.Linear(self.h_dim,self.h_dim))
            Linear_users.append(nn.ReLU())
        self.Linear_user = nn.ModuleList(Linear_users)
        Linear_items = [nn.Linear(m,self.h_dim)]

        for i in range(self.layers):
            Linear_items.append(nn.Linear(self.h_dim, self.h_dim))
            Linear_items.append(nn.ReLU())
        self.Linear_item = nn.ModuleList(Linear_items)
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





