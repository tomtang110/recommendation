import numpy as np
import math
def cal_precision_at_k(k, rankedlist, test_matrix):
    test_set = set(test_matrix)
    rank_set = set(rankedlist)
    hit = len(test_set & rank_set)
    return float(hit / k)
def cal_Recall_at_k_for_each_user(k, rankedlist, test_matrix):
    test_set = set(test_matrix)
    rank_set = set(rankedlist)
    hit = len(test_set & rank_set)
    return float(hit / len(test_set))


def cal_ndcg_at_k_for_each_user(k, rankedlist, testlist):
    idcg_k = 0
    dcg_k = 0
    if len(testlist) < k: k = len(testlist)
    for i in range(k):
        idcg_k += 1 / np.log(i + 2, 2)

    s = set(testlist)
    hits = [idx for idx, val in enumerate(rankedlist) if val in s]
    count = len(hits)
    for i in range(count):
        dcg_k += 1 / math.log(hits[i] + 2, 2)

    return float(dcg_k / idcg_k)
