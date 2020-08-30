# -*- coding: utf-8 -*-
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import numpy as np
import torch.nn as nn
import torch


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


def numerical_stability_softmax(score, dim):

    max_score, _ = score.max(dim, keepdim=True)

    stable_score = score - max_score
    stable_exp = torch.exp(stable_score)
    stable_prob = stable_exp / stable_exp.sum(dim, keepdim=True)

    return stable_prob


def numerical_stability_masked_softmax(vec, mask, dim=1, epsilon=1e-6):

    masked_vec = vec * mask.float()
    max_vec, _ = masked_vec.max(dim, keepdim=True)
    stable_vec = vec - max_vec
    stable_exps = torch.exp(stable_vec)
    masked_exps = stable_exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    masked_prob = masked_exps / masked_sums

    return masked_prob


def numerical_stability_inner_masked_softmax(vec, mask, dim=1, num_phrases=2, topN=10, epsilon=1e-8):

    mask = mask.float()

    if dim==0:
        vec = vec.permute(1,0)
        mask = mask.permute(1,0)

    masked_inner_vec = vec * mask
    masked_inner_vec = masked_inner_vec.contiguous().view(-1, topN)
    inner_mask = mask.contiguous().view(-1, topN)
    inner_max_vec, _ = masked_inner_vec.max(1, True)
    stable_inner_vec = masked_inner_vec - inner_max_vec
    stable_inner_exps = torch.exp(stable_inner_vec)
    masked_inner_exps = stable_inner_exps * inner_mask.float()
    masked_inner_sums = masked_inner_exps.sum(1, keepdim=True) + epsilon
    masked_inner_prob = masked_inner_exps / masked_inner_sums  ## (np*N*np)*N

    masked_inner_vec_total = masked_inner_vec.sum(1).contiguous().view(num_phrases*topN, num_phrases)
    inner_mask_total = inner_mask.sum(1).contiguous().view(num_phrases*topN, num_phrases).ge(1).float()
    masked_inner_vec_total = masked_inner_vec_total * inner_mask_total
    inner_max_vec_total, _ = masked_inner_vec_total.max(1, True)
    stable_inner_exps_total = torch.exp(masked_inner_vec_total-inner_max_vec_total)
    masked_inner_exps_total = stable_inner_exps_total * inner_mask_total
    masked_inner_sum_total = masked_inner_exps_total.sum(1, keepdim=True) + epsilon ## (np*N)*np
    masked_inner_prob_total = masked_inner_exps_total / masked_inner_sum_total
    masked_inner_prob_total = masked_inner_prob_total.contiguous().view(-1).unsqueeze(1) ## (np*N*np) *1

    masked_inner_prob = masked_inner_prob * masked_inner_prob_total
    masked_inner_prob = masked_inner_prob.contiguous().view(num_phrases*topN, num_phrases*topN)

    if dim == 0:
        masked_inner_prob = masked_inner_prob.permute(1,0)

    return masked_inner_prob




def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

if __name__ == '__main__':

    import numpy as np
    relation_conn = [[0,1],[0,2]]
    topN = 10
    conn_map = np.zero(30, 30)

    random_matrix = np.random.random((10,10))
    for rel in relation_conn:
        conn_map[rel[0]*topN:(rel[0]+1)*topN, rel[1]*topN:(rel[1]+1)*topN] = random_matrix
