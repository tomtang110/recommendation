import torch
from torch import nn
from torch import optim
from torch import functional as F

class FFM(nn.Module):

    def __init__(self,m,n,k,f_size,lr,epoch,feature2field):
        self.m = m
        self.n = n
        self.f_size = f_size
        self.lr = lr
        self.epoch = epoch
        self.feature2field = feature2field
        self.k = k

        self.Linear_layer = nn.Linear(n,1)

        self.vector = torch.randn((n,f_size,k),requires_grad=True)

    def forward(self, X):
        self.linear_ = self.Linear_layer(X)
        self.field_aware_interaction = torch.tensor(0.0)
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.field_aware_interaction += torch.mul(
                    torch.mul(self.vector[i,self.feature2field[i]],
                              self.vector[j,self.feature2field[j]]).sum(),
                    torch.mul(self.X[:,i],self.X[:,j])
                )
        out = self.linear_ + self.field_aware_interaction.sum()
        return out

    def train_batch(self,X,Y):
        optimizer = optim.SGD(self.parameters(),lr=self.lr)
        for _ in range(self.epoch):
            optimizer.zero_grad()
            pred = self.forward(X)
            loss = F.BCEWithLogitsLoss(pred,Y)
            loss.backward()
            loss.step()










