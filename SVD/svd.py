import torch
from torch import nn


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
        self.num_feature=config.h_dim    #num of laten features
        self.userNo=config.m               #user num
        self.itemNo=config.n              #item num
        self.bi=nn.Parameter(torch.rand(self.itemNo,1))    #parameter
        self.bu=nn.Parameter(torch.rand(self.userNo,1))    #parameter
        self.U=nn.Parameter(torch.rand(self.num_feature,self.userNo))    #parameter
        self.V=nn.Parameter(torch.rand(self.itemNo,self.num_feature))    #parameter

    def mf_layer(self,train_set=None):
        # predicts=all_mean+self.bi+self.bu.t()+pt.mm(self.V,self.U)
        self.Pre =self.bi + self.bu.t() + torch.mm(self.V, self.U)
        return self.Pre

    def forward(self, train_set):
        output=self.mf_layer(train_set)
        return output