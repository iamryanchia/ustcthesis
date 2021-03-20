import torch
from math import pow

from effdet.object_detection import FasterRcnnBoxCoder, BoxList, IouSimilarity

box_coder = FasterRcnnBoxCoder()
iou_compute = IouSimilarity().compare


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class LIoULoss(object):

    hard_mining = True

    def __init__(self, beta=1.0, alpha=2.0, gamma=8.0, eps=1e-6):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.c = LIoULoss.compute_c(gamma) if self.hard_mining else None

        # avoid underflow
        self.min_iou = pow(eps, 1.0 / gamma)
        self.max_iou = pow(1 - eps, 1.0 / gamma)

    @staticmethod
    def compute_c(gamma):
        """c is used to make c - (1 - iou ^ gamma) * iou >= 0.
        In other words, c is maximum value of (1 - iou ^ gamma) * iou.
        For efficiency, c is precomputed then pass to loss func.
        """

        c = gamma / pow(gamma + 1, 1 / gamma + 1)
        return c

    def __call__(self,
                 encode_pred,
                 encode_target_and_encode_base,
                 weights=None,
                 delta=0.1,
                 size_average=True):
        assert encode_target_and_encode_base.dim(
        ) == 5 and encode_target_and_encode_base.size(-1) == 2

        encode_target_and_encode_base = encode_target_and_encode_base.view(
            -1, 4, 2)
        encode_target, encode_base = encode_target_and_encode_base[
            ..., 0], BoxList(encode_target_and_encode_base[..., 1])
        encode_pred = encode_pred.reshape(-1, 4)

        decode_pred = box_coder.decode(encode_pred, encode_base)
        decode_target = box_coder.decode(encode_target, encode_base)

        # ious = iou_compute(decode_pred, decode_target)
        ious = bbox_overlaps(decode_pred.boxes(),
                             decode_target.boxes(),
                             is_aligned=True,
                             eps=self.eps)

        # smooth l1 loss
        diff = torch.abs(encode_pred - encode_target)
        smooth_l1_loss = torch.where(diff < delta, 0.5 * diff * diff / delta,
                                     diff - 0.5 * delta).sum(1)

        if self.hard_mining:
            with torch.no_grad():
                clip_ious = torch.clip(ious, max=self.max_iou, min=self.min_iou)
                coeff = 1 - torch.pow(clip_ious, self.gamma)
                coeff[ious > self.max_iou] = 0.0
                coeff[ious < self.min_iou] = 1.0
            loss = self.alpha * (self.c - coeff * ious) + smooth_l1_loss
        else:
            loss = 1 - ious + smooth_l1_loss

        if weights is not None:
            loss *= weights.view(-1, 4).type(torch.float32).mean(-1)
        if size_average:
            return loss.mean()
        return loss.sum()
