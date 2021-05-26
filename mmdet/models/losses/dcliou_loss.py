from math import pi, pow

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def liou_loss(encode_decode_preds,
              encode_decode_targets,
              beta=1.0,
              alpha=2.0,
              gamma=8.0,
              c=None,
              hard_mining=False,
              eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets

    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)

    # smooth l1 loss
    diff = torch.abs(encode_pred - encode_target)
    smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                                 diff - 0.5 * beta).sum(1)

    if hard_mining:
        with torch.no_grad():
            coeff = 1 - torch.pow(ious, gamma)
        if c is None:
            c = LIoULoss.compute_c(gamma)
        loss = alpha * (c - coeff * ious) + smooth_l1_loss
    else:
        loss = 1 - ious + smooth_l1_loss
    return loss


def _diou_loss_core(pred, target, eps=1e-6):
    ious = bbox_overlaps(pred, target, is_aligned=True, eps=eps)

    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_diagonal = torch.pow(enclose_wh, 2).sum(1) + eps

    coord_diff = pred - target
    distance = torch.pow(coord_diff[:, :2] + coord_diff[:, 2:], 2).sum(1) / 4.0

    loss = 1 - ious + distance / enclose_diagonal
    return loss, ious


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def diou_loss(pred, target, eps=1e-6):
    loss, _ = _diou_loss_core(pred, target, eps=eps)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ciou_loss(pred, target, eps=1e-6):
    diou_loss, ious = _diou_loss_core(pred, target, eps=1e-6)

    wh_pred = (pred[:, 2:] - pred[:, :2]).clamp(min=eps)
    wh_target = (target[:, 2:] - target[:, :2]).clamp(min=eps)

    v = (4 / pi**2) * torch.pow(
        torch.atan(wh_pred[:, 0] / wh_pred[:, 1]) -
        torch.atan(wh_target[:, 0] / wh_target[:, 1]), 2)
    with torch.no_grad():
        alpha = v / (1 - ious + v).clamp(
            min=eps)  # * torch.pow(wh_pred, 2).sum(1)

    loss = diou_loss + alpha * v
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def navie_iou_loss(pred, target, eps=1e-6):
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = 1 - ious
    return loss


@LOSSES.register_module()
class LIoULoss(nn.Module):

    def __init__(self,
                 beta=1.0,
                 alpha=2.0,
                 gamma=8.0,
                 hard_mining=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(LIoULoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.hard_mining = hard_mining
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.c = LIoULoss.compute_c(gamma) if hard_mining else None

    @staticmethod
    def compute_c(gamma):
        """c is used to make c - (alpha - iou) ^ gamma * iou >= 0.
        In other words, c is maximum value of (alpha - iou) ^ gamma * iou.
        For efficiency, c is precomputed then pass to loss func.
        """
        c = gamma / pow(gamma + 1, 1 / gamma + 1)
        return c

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * liou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
            c=self.c,
            hard_mining=self.hard_mining,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class DIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class CIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class NavieIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(NavieIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * navie_iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
