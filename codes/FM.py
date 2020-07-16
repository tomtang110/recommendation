import torch
from torch import nn
from torch import optim
from torch import functional as F

class FM(nn.Module):
    def __init__(self,epoch,n,k,lr):
        self.lr = lr
        self.embedding = torch.randn((n,k),requires_grad=True)
        self.linear_layer = nn.Linear(n,1)
        self.epoch = epoch
    def forward(self,x):
        out_1 = torch.einsum('mn,nk->mk',x,self.embedding).pow(2).sum(1,keepdim=True)
        out_2 = torch.matmul(x.pow(2),self.embedding.pow(2)).sum(1,keepdim=True)
        out_iner = 1/2*(out_1-out_2)
        return out_iner + self.linear_layer(x)
    def train_batch(self,x,y):
        optimizer = optim.SGD(self.parameters(),lr=self.lr)

        for _ in range(self.epoch):
            optimizer.zero_grad()
            res = self.forward(x)
            loss = F.BCEWithLogitsLoss(res,y)
            loss.backward()
            loss.step()



