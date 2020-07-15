import numpy as np
from collections import defaultdict
class UserCF(object):
    def __init__(self,user_,item_,userIIF=False):
        # 按照index, 一个user 对应一个 item
        # 假设user的index是0-N
        # 假设item的index是0-M
        self.user_ = user_
        self.item_ = item_
        self.userIIF = userIIF

    def userSimilarity(self):

        item_user = defaultdict(set)
        user_item = defaultdict(set)
        for i in range(len(self.user_)):
            user = self.user_[i]
            item = self.item_[i]
            item_user[item].add(user)
            user_item[user].add(item)

        C = np.zeros([len(self.user_), len(self.user_)])
        N = defaultdict(int)
        # C:输出用户u与v共同的物品数目矩阵
        for item,users in item_user.items():
            for user in users:
                N[user] += 1
                for v in users:
                    if user == v:
                        continue
                    if not self.userIIF:
                        C[user][v] += 1
                    else:
                        C[user][v] += 1/np.log(1+len(users))
        for u in range(len(C)):
            for v in range(len(C)):
                C[u][v] /= np.sqrt((N[u] *N[v]))
        return C,user_item

    def recommend(self,userID,K,N):
        W, user_item = self.userSimilarity()
        rank = defaultdict(int)
        iteracted_item = user_item[userID]
        for v in np.argsort(W[userID],reverse=True)[:K]:
            for item in user_item[v]:
                if item not in iteracted_item:
                    rank[item] += W[userID][v]
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]