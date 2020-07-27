import torch
from torch import nn
from torch import functional as F
from torch import optim

class Config(object):
    def __init__(self):
        self.model_name = 'SVD++'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m = 943
        self.n = 1682
        self.h_dim = 256
        self.lambda1 = 0.1
        self.beta1 = 0.1
        self.epoch = 2000
        self.lr= 1e-3
        self.save_path = './Model_save/'
        self.require_improvement = 100


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.p_w = nn.Parameter(torch.randn((self.config.m,self.config.h_dim)))
        self.q_w = nn.Parameter(torch.randn((self.config.h_dim,self.config.n)))
        self.p_bias = nn.Parameter(torch.randn((1,self.config.n)))
        self.q_bias = nn.Parameter(torch.randn((self.config.m,1)))

    def forward(self,R):
        # R shape: [user,item]
        mask = (R >0).float().cuda()
        self.total_n = torch.sum(mask)
        u = torch.sum(R) / torch.sum(mask)
        latent = torch.einsum('mn,hn -> mh',mask,self.q_w) / torch.sqrt(mask.sum(axis=1).view(-1,1))
        Pre = torch.einsum('mh,hn -> mn',(self.p_w + latent),self.q_w)
        self.Pre = self.p_bias + Pre + self.q_bias + u


        regularizes = self.config.lambda1*\
                      (self.config.n * (self.q_bias.norm(p=2).sum() + self.q_w.norm(p=2).sum()) \

                                           +self.config.m * (self.p_bias.norm(p=2).sum() + self.p_w.norm(p=2).sum()) + latent.norm(p=2).sum())
        return Pre,regularizes
