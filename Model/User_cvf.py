import numpy as np
from collections import defaultdict
class Model(object):
    def __init__(self,user_,item_,userIIF=False):
        # 按照index, 一个user 对应一个 item
        # 假设user的index是0-N
        # 假设item的index是0-M
        self.user_ = user_
        self.item_ = item_
        self.userIIF = userIIF
        self.W, self.user_item = self.userSimilarity()

    def userSimilarity(self):

        item_user = defaultdict(set)
        user_item = defaultdict(set)
        for i in range(len(self.user_)):
            user = self.user_[i]
            item = self.item_[i]
            item_user[item].add(user)
            user_item[user].add(item)

        C = defaultdict(dict)
        N = defaultdict(int)
        # C:输出用户u与v共同的物品数目矩阵
        for item,users in item_user.items():
            for user in users:
                N[user] += 1
                for v in users:
                    if user == v:
                        continue
                    C[user][v] = C[user].get(v,0)
                    if not self.userIIF:
                        C[user][v] += 1
                    else:
                        C[user][v] += 1/np.log(1+len(users))
        W = C.copy()
        for u,related_users in C.items():
            for v,cuv in related_users.items():
                W[u][v] = cuv/np.sqrt(N[u]*N[v])

        return W,user_item

    def recommend(self,userID,K,N):

        rank = defaultdict(int)
        iteracted_item = self.user_item[userID]
        for v,wuv in sorted(self.W[userID].items(),key= lambda x:x[1],reverse=True)[:K]:
            for item in self.user_item[v]:
                if item not in iteracted_item:
                    rank[item] += wuv
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]