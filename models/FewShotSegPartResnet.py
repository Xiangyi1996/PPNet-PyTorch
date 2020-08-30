#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from collections import OrderedDict
from util.kmeans import KmeansClustering
# from .vgg import Encoder
from .ResNetBackbone import resnet50
from .Aspp import _ASPP
import numpy as np

class FewShotSegPart(nn.Module):
    """
    Fewshot Segmentation model
    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, pretrained_path=None, cfg=None):
        super().__init__()
        self.GLOBAL_CONST = 0.8
        self.config = cfg #self.config = cfg #

        # Encoder
        # self.encoder = nn.Sequential(OrderedDict([('backbone', Encoder(in_channels)),]))
        self.encoder = resnet50(cfg=cfg)
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=10, init='random')


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
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
        img_fts = self.encoder(imgs_concat) #X*512*53*53
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(torch.device('cuda'))
        outputs = []

        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], 1)
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes #2, 5*512 ; p5*512

            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes] #3, 1*53*53

            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))

            ###### Prototype alignment loss ######
            if self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi


        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        output_semantic = None

        return output, output_semantic, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        # self.dist_part_list_iter.append(F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler) #5*53*53
        # dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1).max(0)[0] * scaler #53*53
        # return dist.unsqueeze(0)

        dist = F.cosine_similarity(fts.unsqueeze(1), prototype[..., None, None].unsqueeze(0), dim=2).max(1)[0] * scaler
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


    def kmeansPrototype(self, fg_fts, bg_fts):

        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts]
        fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc)

        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)

        fg_propotypes = [fg_c + self.GLOBAL_CONST * fg_g for (fg_c, fg_g) in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.GLOBAL_CONST * bg_prop_glo
        return fg_propotypes, bg_propotypes  ## 2, 5*512; 5*512


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


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
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
        qry_prototypes = self.getAlignProto(qry_fts=qry_fts, pred_mask=pred_mask, skip_ways=skip_ways, image_size=(h, w))

        # Compute the support loss
        loss = torch.zeros(1).to(torch.device('cuda'))
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear', align_corners=True)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss

    def getAlignProto(self, qry_fts, pred_mask, skip_ways, image_size):

        """
        qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
        pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H' x W'
        """
        # import ipdb;
        # ipdb.set_trace()

        pred_mask_global = pred_mask.unsqueeze(2)  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes_global = torch.sum(qry_fts.unsqueeze(1) * pred_mask_global, dim=(0, 3, 4))
        qry_prototypes_global = qry_prototypes_global / (pred_mask_global.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        n, c, h, w = qry_fts.shape
        qry_fts_s4 = F.interpolate(input=qry_fts, size=(h * 2, w * 2), mode='bilinear', align_corners=True)  ## N*C*(2h)*(2w)
        qry_fts_s0 = F.interpolate(input=qry_fts, size=image_size, mode='bilinear', align_corners=True)  ## # N x C x H x W
        qry_fts_s4_reshape = qry_fts_s4.permute(0, 2, 3, 1).view(-1, c).contiguous()  ## N*C*2h*2W -> M*C
        qry_fts_s0_reshape = qry_fts_s0.permute(0, 2, 3, 1).view(-1, c).contiguous()  ## N*C*H*W -> L*C
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
        bg_prototypes += qry_prototypes_global[[0]] * self.GLOBAL_CONST

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
            fg_pro += qry_prototypes_global[[way_id+1]] * self.GLOBAL_CONST

            fg_prototypes.append(fg_pro)

        prototypes = [bg_prototypes,] + fg_prototypes    ## [5*d,5*d,5*d]

        return prototypes



