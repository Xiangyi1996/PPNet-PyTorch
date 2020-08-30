#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



def alignLossWithPseudoLabels(qry_fts, pred, supp_fts, fore_mask, back_mask, un_fts, un_plabels, getAlignProto):
    """
    Compute the loss for the prototype alignment branch

    Args:
        qry_fts: embedding features for query images
            expect shape: N x C x H' x W'
        pred: predicted segmentation score
            expect shape: N x (1 + Wa) x H x W
        supp_fts: embedding features for support images
            expect shape: Wa x Sh x C x H' x W'
        fore_mask: foreground masks for support images
            expect shape: way x shot x H x W
        back_mask: background masks for support images
            expect shape: way x shot x H x W

        un_fts: unlabel_images feature map, way*nu*512*h*w
        un_plabels: [nu*h*w, nu*h*w] the pseudo label
        un_weights: [nu*h*w, nu*h*w] weight confidence weights

    """
    n_ways, n_shots, h, w = fore_mask.shape

    # Mask and get query prototype
    pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
    binary_masks = [pred_mask == i for i in range(1 + n_ways)]
    skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
    pred_mask = torch.cat(binary_masks, dim=1).float()  # N x (1 + Wa) x H' x W'
    qry_prototypes = getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))

    # binary_masks = [(pred>=i*5)*(pred<(i+1)*5) for i in range(1 + n_ways)]
    # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
    # pred_mask = [b_mask.float()*pred-nid*5 for nid, b_mask in enumerate(binary_masks)]   ## three masks, indicate the knt location
    # pred_mask = torch.cat(pred_mask, dim=1).float()  # N x (1 + Wa) x H' x W'
    # qry_prototypes = getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))

    # Compute the support loss
    loss = 0
    loss_pseudo = 0

    for way in range(n_ways):

        if way in skip_ways:
            continue
        # Get the query prototypes
        prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]

        img_fts = supp_fts[way]
        supp_dist = [calDistMax(img_fts, prototype) for prototype in prototypes]
        supp_pred = torch.stack(supp_dist, dim=1)
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
        # Construct the support Ground-Truth segmentation
        supp_label = torch.full_like(fore_mask[way], 255, device=img_fts.device).long()
        supp_label[fore_mask[way] == 1] = 1
        supp_label[back_mask[way] == 1] = 0
        # Compute Loss

        # if cfg.SOLVER.FOCAL_LOSS:
        #     pos_pro = (supp_label ==1).float().sum().cpu().numpy() / ((supp_label<=1).long().cpu().numpy().sum() + 1e-6)
        #     loss = loss + FocalLoss(2, pos_pro, True)(supp_pred, supp_label) / n_ways
        # else:
        loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_ways

        un_fts_way = un_fts[way]
        un_plabel_way = un_plabels[way]  ## nu*h*w
        un_dists = [calDistMax(un_fts_way, prototype) for prototype in prototypes]
        un_pred = torch.stack(un_dists, dim=1)  ## bs*2*53*53
        # if cfg.SOLVER.FOCAL_LOSS:
        #     pos_pro = (un_plabel_way==1).float().sum().cpu().numpy() / ((un_plabel_way<=1).float().cpu().numpy().sum() + 1e-6)
        #     loss_pseudo = loss_pseudo + FocalLoss(2,pos_pro,True)(un_pred, un_plabel_way.long()) / n_ways
        # else:
        loss_pseudo = loss_pseudo + F.cross_entropy(un_pred, un_plabel_way.long(), ignore_index=255) / n_ways

    return loss, loss_pseudo


def alignLoss(qry_fts, pred, supp_fts, fore_mask, back_mask, getAlignProto):
    """
    Compute the loss for the prototype alignment branch

    Args:
        qry_fts: embedding features for query images
            expect shape: N x C x H' x W'
        pred: predicted segmentation score
            expect shape: N x (1 + Wa) x H x W
        supp_fts: embedding features for support images
            expect shape: Wa x Sh x C x H' x W'
        fore_mask: foreground masks for support images
            expect shape: way x shot x H x W
        back_mask: background masks for support images
            expect shape: way x shot x H x W
    """
    n_ways, n_shots, h, w = fore_mask.shape

    # Mask and get query prototype
    pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
    binary_masks = [pred_mask == i for i in range(1 + n_ways)]
    skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
    pred_mask = torch.cat(binary_masks, dim=1).float()  # N x (1 + Wa) x H' x W'

    qry_prototypes = getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))

    # Compute the support loss
    loss = 0

    for way in range(n_ways):
        if way in skip_ways:
            continue
        # Get the query prototypes
        prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
        img_fts = supp_fts[way]
        supp_dist = [calDistMax(img_fts, prototype) for prototype in prototypes]
        supp_pred = torch.stack(supp_dist, dim=1)
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
        # Construct the support Ground-Truth segmentation
        supp_label = torch.full_like(fore_mask[way], 255, device=img_fts.device).long()
        supp_label[fore_mask[way] == 1] = 1
        supp_label[back_mask[way] == 1] = 0
        # Compute Loss
        loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_shots / n_ways
    return loss

def alignLossU2S(un_pts, supp_fts, fore_mask, back_mask):
    """
    Compute the loss for the prototype alignment branch

    Args:
        qry_fts: [bg_pt, fg_pt, fg_pt]
        supp_fts: embedding features for support images
            expect shape: Wa x Sh x C x H' x W'
        fore_mask: foreground masks for support images
            expect shape: way x shot x H x W
        back_mask: background masks for support images
            expect shape: way x shot x H x W
    """
    n_ways, n_shots, h, w = fore_mask.shape

    # Compute the support loss
    loss = 0
    for way in range(n_ways):
        # Get the unlabel prototypes
        prototypes = [un_pts[0], un_pts[way + 1]]
        img_fts = supp_fts[way]
        supp_dist = [calDistMax(img_fts, prototype) for prototype in prototypes]
        supp_pred = torch.stack(supp_dist, dim=1)
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
        # Construct the support Ground-Truth segmentation
        supp_label = torch.full_like(fore_mask[way], 255, device=img_fts.device).long()
        supp_label[fore_mask[way] == 1] = 1
        supp_label[back_mask[way] == 1] = 0
        # Compute Loss
        loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_shots / n_ways
    return loss





def getDiscriminativeLoss(un_bg_pts, un_fg_pts, supp_fts, unlabeled_fts, fg_pts, bg_pt, fore_mask, back_mask):

    """
    supp_fts:  Wa x Sh x C x H' x W'
    un_fts_epi: Wa*5*512*53*53
    un_bg_pts: [knt*512, knt*512]
    un_fg_pts: [knt*512, knt*512]

    cosine embedding
    """
    img_fts = torch.cat((supp_fts, unlabeled_fts), dim=1)
    # dis_un_loss1 = torch.clamp(cosineEmbedding(img_fts[0], un_fg_pts[1]), min=0).mean() ## bs*knt*53*53
    # dis_un_loss2 = torch.clamp(cosineEmbedding(img_fts[1], un_fg_pts[0]), min=0).mean()

    ## cosine similarity
    # dis_loss1 = torch.clamp(cosineEmbedding(img_fts[0], fg_pts[1]), min=0).mean()
    # dis_loss2 = torch.clamp(cosineEmbedding(img_fts[1], fg_pts[0]), min=0).mean()


    fore_mask = F.interpolate(fore_mask, size=supp_fts.shape[-2:], mode='nearest')
    back_mask = F.interpolate(back_mask, size=supp_fts.shape[-2:], mode='nearest')

    dis_loss3 = torch.clamp(cosineEmbedding(supp_fts[0], bg_pt)*fore_mask[0].unsqueeze(1), min=0).mean()
    dis_loss4 = torch.clamp(cosineEmbedding(supp_fts[1], bg_pt)*fore_mask[1].unsqueeze(1), min=0).mean()

    # loss = (dis_loss1 + dis_loss2 + dis_loss3 + dis_loss4) / 4
    loss = (dis_loss3 + dis_loss4) / 2
    return loss


def getAlignLossU2S(un_bg_pts, un_fg_pts, supp_fts, fore_mask, back_mask):
    """
    fore_mask:  Wa x Sh x H x W
    back_mask:  Wa x Sh x H x W
    supp_fts:   Wa x Sh x C x H' x W'
    un_bg_pts:  [knt*512, knt*512]
    un_fg_pts:  [knt*512, knt*512]
    supp_masks: Wa x Sh x B x H' x W'

    """

    n_ways, n_shots = supp_fts.shape[:2]
    loss = 0
    for way in range(n_ways):
        prototypes = [un_bg_pts, un_fg_pts[way]]
        img_fts = supp_fts[way] ## s*c*h'*w'
        supp_dist = [calDistMax(img_fts, prototype) for prototype in prototypes] ## [s*h'*w', s*h'*w']
        supp_pred = torch.stack(supp_dist, dim=1)  ## s*2*h'*w'
        supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
        supp_label = torch.full_like(fore_mask[way], 255, device=img_fts.device).long() ## Sh x 2 x H x W
        supp_label[fore_mask[way] == 1] = 1
        supp_label[back_mask[way] == 1] = 0

        if cfg.SOLVER.FOCAL_LOSS:
            pos_pro = (supp_label == 1).float().sum().cpu().numpy() / ((supp_label <= 1).float().cpu().numpy().sum() + 1e-6)
            loss = loss + FocalLoss(2, pos_pro, True)(supp_pred, supp_label.long()) / n_ways
        else:
            loss = loss + F.cross_entropy(supp_pred, supp_label, ignore_index=255) / n_ways

    return loss


def cosineEmbedding(fts, prototype):
    """

    fts: bs*512*53*53
    prototype: knt*512
    dist: bs*knt*53*53
    """
    dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2)
    return dist

def calDistMax(fts, prototype, scaler=20):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: b*512*53*53
        prototype: prototype of one semantic class
            expect shape: 5*512
        return: b*53*53
    """

    dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler

    return dist

