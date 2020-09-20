import pandas as pd
import numpy as np
from collections import defaultdict
class ContentBased(object):
    def __init__(self,rating_file,item_file):
        self.moviesDF =  pd.read_csv(item_file, index_col=None, sep='::',
                                     header=None, names=['movie_id', 'title', 'genres'])
        self.ratingsDF = pd.read_csv(rating_file, index_col=None, sep='::', header=None,
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.item_cate, self.cate_item = self.get_item_cate()
        self.up = self.get_up()
    def get_item_cate(self, topK=10):
        movie_rating_avg = self.ratingsDF.groupby('movie_id')['rating'].agg(
            {'item_ratings_mean': np.mean}).reset_index()
        items = movie_rating_avg['movie_id'].values
        scores = movie_rating_avg['item_ratings_mean'].values
        item_score_veg = defaultdict(float)
        for item,score in zip(items,scores):
            item_score_veg[item] = score

        item_cate = defaultdict(dict)
        items = self.moviesDF['movie_id'].values
        genres = self.moviesDF['genres'].apply(lambda x:x.split('|')).values
        for item,genres_list in zip(items,genres):
            ratio = 1/len(genres_list)
            for genre in genres_list:
                item_cate[item][genre] = ratio
        recode = defaultdict(dict)
        for item in item_cate:
            for genre in item_cate[item]:
                recode[genre][item] = item_score_veg[item]
        # 这个类型前十的电影
        cate_item = defaultdict(list)
        for cate in recode:
            for zuhe in sorted(recode[cate].items(), key=lambda x: x[1], reverse=True)[::topK]:
                cate_item[cate].append(zuhe[0])

        return item_cate, cate_item
    def get_time_score(self,timestamp,fix_time_stamp):
        total_sec = 24*60*60
        delta = (fix_time_stamp-timestamp)/total_sec/100
        return (1/(1+delta),3)
    def get_up(self,score_thr = 4.0,topK=5):
        ratingsDF = self.ratingsDF[self.ratingsDF['rating'] > score_thr]
        fix_time_stamp = ratingsDF['timestamp'].max()
        ratingsDF['time_score'] = ratingsDF['timestamp'].apply(lambda x: self.get_time_score(x,fix_time_stamp))

        users = ratingsDF['user_id'].values
        items = ratingsDF['movie_id'].values
        ratings = ratingsDF['rating'].values
        scores = ratingsDF['time_score'].values

        recode = defaultdict(dict)
        up = defaultdict(list)

        for userid, itemid, rating, time_score in zip(users,items,ratings,scores):
            for cate in self.item_cate[itemid]:
                recode[userid][cate] += rating*time_score*self.item_cate[itemid][cate]


        for userid in recode:
            total_score = 0
            for zuhe in sorted(recode[userid].items(),key=lambda x:x[1],reverse=True)[::topK]:
                up[userid].append((zuhe[0],zuhe[1]))
                total_score += zuhe[1]
            for index in range(len(up[userid])):
                up[userid][index] = (up[userid][index][0], round(up[userid][index][1] / total_score, 3))
        return up

    def recommend(self,userID,K=10):
        if userID not in self.up:return
        recom_res = defaultdict(list)
        for zuhe in self.up[userID]:
            cate,ratio = zuhe
            num = int(K*ratio) + 1
            if cate not in self.cate_item:continue
            rec_list = self.cate_item[cate][:num]
            recom_res[userID] += rec_list
        return recom_res


