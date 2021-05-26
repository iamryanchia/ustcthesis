_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_coco.py',
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))
