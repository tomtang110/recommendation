import torch
from torch import nn
from torch import functional as F
from torch import optim
class Svdpp(nn.Module):
    def __init__(self,m,n,hidden_dim,alpha,lambda_1,beta1,epoch):
        self.m = m
        self.n = n
        self.h_dim = hidden_dim
        self.p_w = torch.randn((self.m,self.h_dim),requires_grad=True)
        self.q_w = torch.randn((self.h_dim,self.n),requires_grad=True)
        self.p_bias = torch.randn((1,self.m),requires_grad=True)
        self.q_bias = torch.randn((self.n,1),requires_grad=True)
        self.alpha = alpha
        self.lambda1 = lambda_1
        self.beta1 = beta1
        self.epoch = epoch
    def forward(self,R):
        mask = R >0
        self.total_n = torch.sum(mask)
        u = torch.sum(R) / torch.sum(mask)
        latent = torch.einsum('mn,hn -> mh',mask,self.q_w) / mask.sum(axis=1)
        Pre = torch.einsum('mh,hn -> mn',(self.p_w + latent),self.q_w)
        Pre = self.p_bias + Pre + self.q_bias + u
        return Pre
    def train_batch(self,R):
        self.para = [self.p_w,self.q_w,self.p_bias,self.q_bias]
        optimizer = optim.SGD(self.para,lr = self.alpha)
        for i in range(self.epoch):
            optimizer.zero_grad()
            Pre = self.forward(R)
            loss = F.mse_loss(Pre,R) + self.lambda1*(self.n*((self.q_bias**2).sum() +(self.q_w ** 2).sum())+
                                                     self.m*((self.p_bias**2).sum()+(self.p_w**2).sum()))/2

            loss.backward()
            optimizer.step()
        return loss


