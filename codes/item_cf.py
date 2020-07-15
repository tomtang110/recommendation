import numpy as np
from collections import defaultdict

class ItemCF(object):
    def __init__(self,users,items,useIUF):
        self.users_ = users
        self.items_ = items
        self.IUF = useIUF
    def ItemSimilarity(self):
        user_item = defaultdict(set)
        for i in range(len(self.users_)):
            user = self.users_[i]
            item = self.items_[i]
            user_item[user].add(item)

        C = np.zeros([len(self.items_), len(self.items_)])
        N = defaultdict(int)

        for u,items in user_item.items():
            for item in items:
                N[item] += 1
                for j in items:
                    if item == j:
                        continue
                    if not self.IUF:
                        C[item][j] += 1
                    else:
                        C[item][j] += 1/np.log(1+len(items))
        for u in range(len(C)):
            for v in range(len(C)):
                C[u][v] /= np.sqrt((N[u] *N[v]))
        return C,user_item

    def recommend(self,userID,K,N):
        W, user_item = self.ItemSimilarity()
        rank =defaultdict(float)
        iteracted_items = user_item[userID]
        for i in iteracted_items:
            for wij in np.argsort(W[i],reverse=True)[:K]:
                rank[wij] += W[i][wij]
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]