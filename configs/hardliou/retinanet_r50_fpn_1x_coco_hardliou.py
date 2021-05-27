_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_coco.py',
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        still_need_encoded_bbox=True,
        loss_bbox=dict(
            type='LIoULoss',
            loss_weight=1.0,
            hard_mining=True,
            alpha=2.0,
            gamma=8.0,
        )))
