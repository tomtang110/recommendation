import torch
from torch import nn
from torch import functional as F
from torch import optim

class Model(nn.Module):

    def __init__(self,m,n,hidden_dim,lambda_value,alpha,epoch):
        self.m = m
        self.n = n

        self.h_dim = hidden_dim
        self.lr = alpha
        self.epoch = epoch
        self.lambda_value = lambda_value
        self.Linear1 = nn.Linear(n,self.h_dim)
        self.Linear2 = nn.Linear(self.h_dim,n)

    def forward(self, X):
        self.mask = X>0

        encoder = self.Linear1(X)
        encoder = F.sigmoid(encoder)
        decoder = self.Linear2(encoder)

        return decoder
    def train_batch(self,X):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        for _ in range(self.epoch):
            optimizer.zero_grad()
            cost = self.forward(X)
            cost = torch.mul(F.mse_loss((X - self.forward(X))),self.mask)
            pre_reg_cost = self.Linear1.weight.pow(2) + self.Linear2.weight.pow(2)

            cost = cost + self.lambda_value * 0.5 * pre_reg_cost
            cost.backward()
            optimizer.step()




