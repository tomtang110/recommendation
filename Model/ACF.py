import torch
from torch import nn
from torch import functional as F

class Model(nn.Module):

    def __init__(self,m,n,hid_dim,f_dim):
        super(Model,self).__init__()
        self.m = m
        self.n = n
        self.h_dim = hid_dim
        self.f_dim = f_dim
        self.user_embed = nn.Embedding(m,self.h_dim)
        self.item_embed = nn.Embedding(n,self.h_dim)
        self.item_embed_latent = nn.Embedding(n,self.h_dim)
        self.W_2u = nn.parameter(torch.FloatTensor((self.h_dim,self.f_dim)))
        self.W_1u = nn.parameter(torch.FloatTensor((self.h_dim, 1)))
        self.W_2x = nn.parameter(torch.FloatTensor((self.f_dim,self.f_dim)))
        self.W_1x = nn.parameter(torch.FloatTensor((self.n, self.n)))
        self.W_1v = nn.parameter(torch.FloatTensor((self.h_dim, 1)))
        self.W_1p = nn.parameter(torch.FloatTensor((self.h_dim, 1)))
        self.b2 = nn.parameter(torch.FloatTensor((self.f_dim)))
        self.b1 = nn.parameter(torch.FloatTensor((self.n)))
        self.W2 = nn.parameter(torch.FloatTensor(self.f_dim,self.f_dim))
        self.W1 = nn.parameter(torch.FloatTensor(self.n))
        self.lambdas = 0.5
    def forward(self,users,pos_items,pos_feature,neg_items,neg_feature):
        # pos_items (user,items)
        u = self.user_embed(users)
        pos_it = self.item_embed(pos_items) #(m,n,h_dim)
        neg_it = self.item_embed(neg_items)
        pos_it_mask = pos_items > 0
        neg_it_mask = neg_items >0
        item_it = torch.cat([pos_it,neg_it],axis=0)

        pos_it_latent = self.item_embed_latent(pos_items)
        neg_it_latent = self.item_embed_latent(neg_items)
        item_it_latent = torch.cat([pos_it_latent,neg_it_latent],axis=0)

        item_feature = torch.cat([pos_feature,neg_feature],axis=0)
        b = (torch.matmul(u,self.W_2u) + torch.matmul(item_feature,self.W_2x) + self.b2).mm(self.W2)
        b_pro = F.softmax(b,axis=2)

        b_ = (b * b_pro).sum(axis=2)

        a_= ((torch.matmul(u,self.W_1u) + torch.matmul(item_it,self.W_1v)) + \
                torch.matmul(item_it_latent,self.W_1p) + torch.matmul(b_,self.W_1x) + self.b1).mm(self.W1)
        alpha_pos = F.softmax(a_,axis=1)


        sum_p_pos = u + (alpha_pos * item_it_latent).sum(axis=1)


        loss = -torch.log(F.sigmoid(torch.einsum('md,mnd->mn',sum_p_pos,pos_it)*pos_it_mask - \
               torch.einsum('md,mnd->mn',sum_p_pos,neg_it) * neg_it_mask)).sum()
        regular = self.lambdas * (self.user_embed.weight.norm(p=2).sum()+\
                                  self.item_embed.weight.norm(p=2).sum()+\
                                  self.item_embed_latent.weight.norm(p=2).sum())
        return loss + regular











