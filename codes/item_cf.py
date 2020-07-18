import numpy as np
from collections import defaultdict

class Model(object):
    def __init__(self,users,items,useIUF=True):
        self.users_ = users
        self.items_ = items
        self.IUF = useIUF
    def ItemSimilarity(self):
        user_item = defaultdict(set)
        for i in range(len(self.users_)):
            user = self.users_[i]
            item = self.items_[i]
            user_item[user].add(item)

        C = defaultdict(dict)
        N = defaultdict(int)

        for u,items in user_item.items():
            for item in items:
                N[item] += 1
                for j in items:
                    if item == j:
                        continue
                    C[item][j] = C[item].get(j, 0)
                    if not self.IUF:
                        C[item][j] += 1
                    else:
                        C[item][j] += 1/np.log(1+len(items))
        W = C.copy()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W[i][j] = cij / np.sqrt(N[i] * N[j])

        return W, user_item

    def recommend(self,userID,K,N):
        W, user_item = self.ItemSimilarity()
        rank =defaultdict(float)
        iteracted_items = user_item[userID]
        for i in iteracted_items:
            for j,wij in sorted(W[i].items(),key= lambda x:x[1],reverse=True)[:K]:
                rank[j] += wij
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]