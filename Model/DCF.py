import torch
from torch import nn
from torch import functional as F
from torch import optim

class Model(nn.Module):

    def __init__(self,m,n,userf,itemf,h_dim,alpha,lr,epoch,beta,lambdas,c):

        self.m = m
        self.n = n
        self.userf = userf
        self.itemf = itemf
        self.h_dim = h_dim
        self.alpha = alpha
        self.lr = lr
        self.epoch = epoch
        self.beta = beta

        self.lambdas = lambdas
        self.c = c

        self.w1 = torch.randn((self.userf,self.userf),requires_grad=True)
        self.w2 = torch.randn((self.itemf,self.itemf),requires_grad=True)


        self.p1 = torch.randn((self.userf,self.h_dim),requires_grad=True)
        self.p2 = torch.randn((self.itemf,self.h_dim),requires_grad=True)

        self.w11 = torch.randn((self.userf, self.userf), requires_grad=True)
        self.w22 = torch.randn((self.itemf, self.itemf), requires_grad=True)

    def forward(self,users,items):
        self.U_1 = torch.matmul(users,self.w1)
        U = self.U_1.mm(self.p1)
        self.V_1 = torch.matmul(items,self.w2)
        V = self.V_1.mm(self.p2)

        self.U_2 = self.U_1.mm(self.w11)
        self.V_2 = self.V_1.mm(self.w22)

        return U,V,self.U_2,self.V_2
    def train_batch(self,users,items,R):

        optimizers = optim.SGD([self.w1,self.w2,self.p1,self.p2],lr=self.lr)
        optimizers.zero_grad()
        mask = R > 0
        for _ in range(self.epoch):
            U,V,X_,V_ = self.forward(users,items)
            loss1 = self.alpha*torch.mul(mask,(R-torch.matmul(U,V.view(self.h_dim,-1)))).norm(p=2).sum()

            loss2 = (self.c * users.view(self.userf,-1) - self.w1.mm(X_)).norm(p=2) +\
                    self.lambdas * (self.p1.mm(U.view(self.h_dim,-1))- self.w1.mm(users.view(self.userf,-1))).norm(p=2)

            loss3 = (self.c * items.view(self.itemf, -1) - self.w2.mm(V_)).norm(p=2) + \
                    self.lambdas * (self.p2.mm(V.view(self.h_dim, -1)) - self.w2.mm(items.view(self.userf, -1))).norm(
                p=2)

            loss =loss1 + loss2 + loss3

            loss.backward()
            optimizers.step()




