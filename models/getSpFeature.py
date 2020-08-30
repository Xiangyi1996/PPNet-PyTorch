# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 14:13
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from .ops import Linear, numerical_stability_masked_softmax, masked_softmax, numerical_stability_softmax


def getSpFeats(un_feats, un_segments):

    """
    un_feats: N*2048*53*53
    un_segments: N*1*417*417
    """
    n,c, h, w = un_feats.shape
    un_segments = F.interpolate(un_segments, (h, w), mode='nearest')
    s_feats = []
    for nid in range(n):
        un_seg = un_segments[nid]
        un_feat = un_feats[nid]
        unique = un_seg.unique()
        nid_feats = []
        for uid in unique:
            mask = (un_seg==uid).float()
            feat = (un_feat*mask).sum((1,2))/mask.sum()
            nid_feats.append(feat)
        nid_feats = torch.stack(nid_feats, dim=0) ## 100*2048
        s_feats.append(nid_feats)

    return s_feats


def select_topk_spfeats(s_feats, prototypes, cfg):

    """
    prototypes: [1*2048, 1*2048, 1*2048]  ## num_class
    s_feats: [N*2048, N*2048, xxx] ## num_unlabel
    """

    prototypes = torch.cat(prototypes, dim=0) # p*2048
    topk_sp_for_pt = [[] for _ in range(prototypes.shape[0])]
    for feat in s_feats:
        dist = F.cosine_similarity(feat.unsqueeze(1), prototypes.unsqueeze(0), dim=2) ## N*P
        dist_topk_value, dist_topk_indices = torch.topk(dist.permute(1, 0), dim=1, k=cfg['topk'])

        for pid, (p_value, p_topk) in enumerate(zip(dist_topk_value, dist_topk_indices)):
            # print(p_value)
            #Todo
            gt = torch.where(p_value>cfg['p_value_thres'])[0] ## we just select the postive candidates
            if len(gt)>0.1:
                p_topk = p_topk[:gt[-1]+1]
                topk_sp_for_pt[pid].append(torch.index_select(feat, dim=0, index=p_topk))
            else:
                topk_sp_for_pt[pid].append(None)

    return topk_sp_for_pt


class GraphTransformer(nn.Module):

    def __init__(self, in_channels, out_channels, scale=0.2):
        super(GraphTransformer, self).__init__()

        self.out_channels = out_channels
        Linear = nn.Linear
        self.inner_w1 = Linear(in_channels, out_channels, bias=False)
        # self.inner_w2 = Linear(in_channels, out_channels, bias=False)
        self.inner_trans = Linear(in_channels, in_channels, bias=False)

        self.inter_w1 = Linear(in_channels, out_channels, bias=False)
        self.inter_w2 = Linear(in_channels, out_channels, bias=False)
        # self.inter_trans = Linear(in_channels, in_channels, bias=True)


    def forward(self, topk_feats, prototypes):

        """
        topk_feats: [[],[],[],]  ## num_class, num_unlabel
        prototypes: [5*2048, 5*2048, 5*2048] ## num_class
        :return:
        """


        un_prototypes = []

        for feats, protos in zip(topk_feats, prototypes):
            # self-message propagation in one image
            feats_all = []
            for feat in feats:
                if feat is not None:
                    # feats_all.append(feat)
                    if feat.shape[0] == 1:
                        feats_all.append(feat)
                    else:
                        feat_embed_1 = self.inner_w1(feat)
                        # feat_embed_2 = self.inner_w2(feat)
                        atte = torch.mm(feat_embed_1, feat_embed_1.permute(1, 0))/self.out_channels**0.5
                        atte = numerical_stability_softmax(atte, dim=1)
                        feat = feat + F.relu(self.inner_trans(torch.mm(atte, feat)))
                        feats_all.append(feat)

            if len(feats_all) != 0:
                feats_all = torch.cat(feats_all, dim=0)  ## combine all image together
                feats_all_embed = self.inter_w1(feats_all)
                protos_embed = self.inter_w2(protos)
                atte = torch.mm(protos_embed, feats_all_embed.permute(1, 0))/self.out_channels**0.5 #weight
                atte = numerical_stability_softmax(atte, 1)
                un_protos = torch.mm(atte, feats_all)
                un_prototypes.append(un_protos)
            else:
                fake_un_protos = torch.zeros_like(protos).to(torch.device('cuda'))
                un_prototypes.append(fake_un_protos)

        return un_prototypes











































