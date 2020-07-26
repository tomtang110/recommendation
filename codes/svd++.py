import torch
from torch import nn
from torch import functional as F
from torch import optim

class Config(object):
    def __init__(self):
        self.model_name = 'SVD++'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m = 943
        self.n = 1650
        self.h_dim = 256
        self.lambda1 = 0.1
        self.beta1 = 0.1
        self.epoch = 10
        self.lr= 1e-3


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.p_w = torch.randn((self.config.m,self.config.h_dim),requires_grad=True)
        self.q_w = torch.randn((self.config.h_dim,self.config.n),requires_grad=True)
        self.p_bias = torch.randn((1,self.config.n),requires_grad=True)
        self.q_bias = torch.randn((self.config.m,1),requires_grad=True)

    def forward(self,R):
        # R shape: [user,item]
        mask = R >0
        self.total_n = torch.sum(mask)
        u = torch.sum(R) / torch.sum(mask)
        latent = torch.einsum('mn,hn -> mh',mask,self.q_w) / mask.sum(axis=1).view(-1,1)
        Pre = torch.einsum('mh,hn -> mn',(self.p_w + latent),self.q_w)
        self.Pre = self.p_bias + Pre + self.q_bias + u


        regularizes = self.config.lambda1*\
                      (self.config.n * (self.q_bias.norm(p=2).sum() + self.q_w.norm(p=2).sum()) \

                                           +self.config.m * (self.p_bias.norm(p=2).sum() + self.p_w.norm(p=2).sum()) + latent.norm(p=2).sum())
        return Pre,regularizes
    def train_batch(self,R):
        self.para = [self.p_w,self.q_w,self.p_bias,self.q_bias]
        optimizer = optim.SGD(self.para,lr = self.config.lr)
        for i in range(self.epoch):
            optimizer.zero_grad()
            Pre = self.forward(R)
            loss = F.mse_loss(Pre,R) + self.config.lambda1*(self.config.n*((self.q_bias**2).sum() +(self.q_w ** 2).sum())+
                                                     self.config.m*((self.p_bias**2).sum()+(self.p_w**2).sum()))/2

            loss.backward()
            optimizer.step()
        return loss


