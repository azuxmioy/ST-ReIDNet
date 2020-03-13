from __future__ import absolute_import
import os, sys
import functools
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

class GANLoss(nn.Module):
    def __init__(self, smooth=False):
        super(GANLoss, self).__init__()
        self.smooth = smooth
        self.criterion = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            real_label = random.uniform(0.9,1.0)
            fake_label = random.uniform(0.0,0.1)
        if target_is_real:
            target_tensor = torch.ones_like(input).fill_(real_label)
        else:
            target_tensor = torch.zeros_like(input).fill_(fake_label)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.criterion(input, target_tensor)


class TripletLoss(nn.Module):

    def __init__(self, margin='soft', batch_hard=True, distractor=False):
        super(TripletLoss, self).__init__()
        self.batch_hard = batch_hard
        self.distractor = distractor
        self.margin = margin
        if self.margin == 'soft':
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)


    def __call__(self, global_feat, labels, normalize_feature=False):

        if normalize_feature:
            global_feat = self.normalize(global_feat, axis=-1)
        dist_mat = self.euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = self.hard_example_mining( dist_mat, labels)
       
        if self.distractor:
            n_distractor = dist_ap.size(0) // 3
            dist_ap = dist_ap[:-n_distractor]
            dist_an = dist_an[:-n_distractor]

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin == 'soft':
            diff = torch.clamp(dist_an - dist_ap, min=-50)
            loss = self.ranking_loss(diff, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)

        #print(dist_ap)
        #print(dist_an)
        #print(loss)
        
        return loss, dist_ap, dist_an

    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
        """
        assert len(labels.size()) == 1
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        assert labels.size(0) == N
        #print(labels)
        # shape [N, N]
        #is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        #is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        identity = torch.eye(N).byte()
        #print(identity)
        identity = identity.cuda() if labels.is_cuda else identity
        same_id = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        #print(same_id)
        is_neg = same_id ^ 1
        is_pos = same_id ^ identity
        #print(is_neg)
        #print(is_pos)
        #print(dist_mat)
        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        #dist_ap, relative_p_inds = torch.max(
        #    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        dist_ap, relative_p_inds = torch.max( dist_mat * is_pos.float(), 1, keepdim=True)
        
        
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        #dist_an, relative_n_inds = torch.min(
        #    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)


        if self.batch_hard:
            dist_an, relative_n_inds = torch.min( dist_mat + 1e5 * same_id.float(), 1, keepdim=True)
            dist_ap, relative_p_inds = torch.max( dist_mat * is_pos.float(), 1, keepdim=True)
        else:
            idx = is_pos.topk(k=1, dim=1)[1].view(-1,1)
            dist_ap = torch.gather(dist_mat, dim=1, index=idx)
            idx = is_pos.topk(k=1, dim=1)[1].view(-1,1)
            dist_an = torch.gather(dist_mat, dim=1, index=idx)

        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds and self.batch_hard:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                .copy_(torch.arange(0, N).long())
                .unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(
                ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds

        return dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class MaskedL1loss(nn.Module):

    def __init__(self, use_mask = True, alpha = 9.0):
        super(MaskedL1loss, self).__init__()
        self.alpha = alpha
        self.use_mask = use_mask

    def forward(self, inputs, mask, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            mask : human mask for higher reconstruction loss
            targets: ground truth labels with shape (num_classes)
        """
        global_loss = F.l1_loss(inputs, targets)
        if not self.use_mask:
            return global_loss

        masks = torch.unsqueeze(mask, 1)
        masks = masks.repeat(1,3,1,1)

        mask_loss =  F.l1_loss(inputs*masks, targets*masks)
        loss = (global_loss + self.alpha * mask_loss) / (1 + self.alpha)

        return loss
