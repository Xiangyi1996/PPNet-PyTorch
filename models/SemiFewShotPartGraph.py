#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from util.kmeans import KmeansClustering
from .AlignLossFunctions import alignLoss

from .getSpFeature import GraphTransformer, getSpFeats, select_topk_spfeats
from .ResNetBackbone import resnet50Sem
from .Aspp import _ASPP


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


class SemiFewShotSegPartGraph(nn.Module):
    """
    Semi-Fewshot Segmentation model by using the lots of unlabel images
    """
    def __init__(self, cfg=None):
        super(SemiFewShotSegPartGraph, self).__init__()
        self.config = cfg #self.config = cfg #
        self.encoder = resnet50Sem(cfg=cfg)
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=20, init='random')
        self.n_unlabel = cfg['task']['n_unlabels']
        self.n_ways = cfg['task']['n_ways']
        self.n_shots = cfg['task']['n_shots']
        self.kmean_cnt = self.config['center']
        self.device = torch.device('cuda')
        self.channel = 2048
        self.global_const = 0.8
        self.pt_lambda = self.config['pt_lambda']
        print('pt_lamda: ', self.pt_lambda)
        self.un_bs = cfg['un_bs']
        self.latentGraph = GraphTransformer(in_channels=2048, out_channels=512)
        self.aspp = _ASPP(2048, 16, [6, 12, 18, 24])

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, un_imgs, unlabel_segments=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]


        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts, image_fts_semantic = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        if self.training and self.config['model']['sem']:
            output_semantic = self.aspp(image_fts_semantic)
            output_semantic = F.interpolate(output_semantic, size=img_size, mode='bilinear', align_corners=True)
        else:
            output_semantic = None

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        un_imgs = torch.cat([torch.cat(way, dim=0) for way in un_imgs], dim=0)
        un_mask = torch.cat([torch.cat(way, dim=0) for way in unlabel_segments], dim=0).unsqueeze(1)  # (Wa*Su)x1x417x417
        with torch.no_grad():
            un_fts, _ = self.encoder(un_imgs) #20,2048,53,53

        un_segments_feats = getSpFeats(un_fts, un_mask)

        ###### Compute loss #####
        align_loss = torch.zeros(1).to(self.device)
        align_loss_cs = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(batch_size):

            """get the prototypes from support images"""
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1)
                            for shot in range(n_shots)] for way in range(n_ways)]
            fg_prototypes, bg_prototype, fg_fts, bg_fts = self.kmeansPrototype(supp_fg_fts, supp_bg_fts) #2, 5*512; 5*512

            un_segments_topk = select_topk_spfeats(un_segments_feats, [bg_fts,]+fg_fts, self.config)
            un_segments_topk = [[p for p in func(un_topk, self.un_bs)] for un_topk in un_segments_topk]
            num_class = len(un_segments_topk)
            num_fragments = len(un_segments_topk[0])

            """update the prototypes from the unlabel images"""

            un_pts = [[] for _ in range(num_class)]

            for uid in range(num_fragments):

                feat_topk = [topk[uid] for topk in un_segments_topk]
                un_prototypes = self.latentGraph(feat_topk, [bg_prototype,]+fg_prototypes) #2,bs*5*512; bs*5*512
                bg_prototype = bg_prototype * self.pt_lambda + (1 - self.pt_lambda) * un_prototypes[0]
                fg_prototypes = [fg_pt * self.pt_lambda + (1 - self.pt_lambda)*un_fg_pt for fg_pt, un_fg_pt in zip(fg_prototypes, un_prototypes[1:])]

                for wid, un_pts_fid in enumerate(un_prototypes):
                    un_pts[wid].append(un_pts_fid)

            ## propotoypes in unlabel images,
            un_pts=[torch.stack(pts, dim=0).mean(0) + self.global_const * torch.stack(pts, dim=0).mean((0,1)) for pts in un_pts]

            ###### Compute the distance ######
            prototypes = [bg_prototype, ] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype, take_max=True) for prototype in prototypes] #3, 1*53*53
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))


            ###### Prototype alignment loss ######
            if self.training:
                align_loss_epi = alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi], self.getAlignProto)
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, output_semantic, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20, take_max=False):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: 1*512*53*53
            prototype: prototype of one semantic class
                expect shape: 5*512
            return: 5*53*53
        """
        if take_max:
            dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler
        else:
            dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2) * scaler

        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getFeaturesArray(self, fts, mask, upscale=2):

        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:]
        h, w = mask.shape[1:]

        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C

        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1*upscale, w1*upscale), mode='nearest').view(-1)
        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)  ## l*c
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]]*0  # 1 x C
            else:
                fts = fts[mask1>0]
        else:
            fts = F.interpolate(fts, size=(h1*upscale, w1*upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1*w1*upscale**2, c)
            fts = fts[mask_bilinear>0]

        return (fts, masked_fts)


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def kmeansPrototype(self, fg_fts, bg_fts):

        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc)

        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)
        fg_propotypes = [fg_c + self.global_const*fg_g for (fg_c, fg_g) in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.global_const * bg_prop_glo

        return fg_propotypes, bg_propotypes, fg_prop_glo, bg_prop_glo  ## 2, 5*512; 5*512



    def getAlignProto(self, qry_fts, pred_mask, skip_ways, image_size):

        """
        qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
        pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H' x W'
        """

        pred_mask_global = pred_mask.unsqueeze(2)  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes_global = torch.sum(qry_fts.unsqueeze(1) * pred_mask_global, dim=(0, 3, 4))
        qry_prototypes_global = qry_prototypes_global / (pred_mask_global.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        n, c, h, w = qry_fts.shape
        qry_fts_s4 = F.interpolate(input=qry_fts, size=(h * 2, w * 2), mode='bilinear', align_corners=True)  ## N*C*(2h)*(2w)
        qry_fts_s0 = F.interpolate(input=qry_fts, size=image_size, mode='bilinear', align_corners=True)  ## # N x C x H x W
        qry_fts_s4_reshape = qry_fts_s4.permute(0, 2, 3, 1).contiguous().view(-1, c)  ## N*C*2h*2W -> M*C
        qry_fts_s0_reshape = qry_fts_s0.permute(0, 2, 3, 1).contiguous().view(-1, c)  ## N*C*H*W -> L*C
        pred_mask_s4 = F.interpolate(input=pred_mask, size=(h * 2, w * 2), mode='nearest')
        pred_mask_s0 = F.interpolate(input=pred_mask, size=image_size, mode='nearest')  # N x (1 + Wa) x H' x W'

        num_background = pred_mask_s4[:, 0].sum()
        if num_background == 0 :
            bg_prototypes  = qry_fts_s4_reshape[[0]]*0  ## 1*C
        else:
            if num_background <= 10:
                bg_pred = qry_fts_s0_reshape[pred_mask_s0[:, 0].view(-1)>0]  ## all bg fts
            else:
                bg_pred = qry_fts_s4_reshape[pred_mask_s4[:, 0].view(-1)>0]
            bg_prototypes = self.kmeans.cluster(bg_pred)  ## 5*C

        # print(bg_prototypes.shape, qry_prototypes_global.shape, qry_prototypes_global[[0]].shape)
        bg_prototypes += qry_prototypes_global[[0]] * self.global_const

        pred_mask_s4_ways = pred_mask_s4[:, 1:].permute(1, 0, 2, 3)
        pred_mask_s0_ways = pred_mask_s0[:, 1:].permute(1, 0, 2, 3)

        fg_prototypes = []

        for way_id in range(pred_mask_s0_ways.shape[0]):

            ## pred_mask_w : N*H*W
            if way_id in skip_ways:
                fg_prototypes.append(None)
                continue

            pred_mask_w_s4, pred_mask_w_s0 = pred_mask_s4_ways[way_id], pred_mask_s0_ways[way_id]
            pred_mask_w_s4  = pred_mask_w_s4.view(-1).contiguous()
            pred_mask_w_s0  = pred_mask_w_s0.view(-1).contiguous()

            num_pos = pred_mask_w_s4.sum()
            if num_pos <= 10:
                qry_fts_w = qry_fts_s0_reshape[pred_mask_w_s0>0]
            else:
                qry_fts_w = qry_fts_s4_reshape[pred_mask_w_s4>0] ## M*c

            fg_pro = self.kmeans.cluster(qry_fts_w)
            fg_pro += qry_prototypes_global[[way_id+1]] * self.global_const

            fg_prototypes.append(fg_pro)

        prototypes = [bg_prototypes,] + fg_prototypes    ## [5*d,5*d,5*d]

        return prototypes




