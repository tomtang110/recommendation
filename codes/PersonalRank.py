from collections import defaultdict
class PersonalRank(object):
    def __init__(self,user_id,item_id):
        X,Y = ['user_'+str(i) for i in user_id],['item_'+str(i) for i in item_id]
        self.X = X
        self.Y= Y
        self.G = self.get_graph(X,Y)

    def get_graph(self,X,Y):
        G = defaultdict(dict)
        for i in range(len(X)):
            user = X[i]
            item = Y[i]
            G[item][user] = 1

        for i in range(len(Y)):
            user = X[i]
            item = Y[i]
            G[user][item] = 1
        return G
    def recommend(self,alpha,UserID,max_depth,K=10):
        userID = 'user_' + str(UserID)
        rank = {x:0 for x in self.G.keys()}
        rank[userID] = 1
        for k in range(max_depth):
            tmp = {x:0 for x in self.G.keys()}
            for i, ri in self.G.items():
                for j, _ in ri.items():
                    tmp[j] += alpha * rank[i] / (1*len(ri))
            tmp[userID] += (1-alpha)
            rank = tmp
        lst = sorted(rank.items(),key=lambda x:x[1],reverse=True)
        res = [l[0] for l in lst if l in self.Y][::K]
        return res


