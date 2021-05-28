#!/usr/bin/env bash

set -oe

export CUDA_VISIBLE_DEVICES=3,4,5,6
GPUS_NUM=$((${#CUDA_VISIBLE_DEVICES} / 2 + 1))

# use bifpn (baseline)
# ./distributed_train.sh $GPUS_NUM ../__DATASET/coco --model efficientdet_d2 -b 6 --amp --lr .036 --sync-bn \
#     --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.99995

# add liou loss
# ./distributed_train.sh $GPUS_NUM ../__DATASET/coco --model efficientdet_d2 -b 6 --amp --lr .036 --sync-bn \
#     --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.99995 \
#     --need-encode-base --no-hard-mining

# add PaFPN
# ./distributed_train.sh $GPUS_NUM ../__DATASET/coco --model efficientdet_d2 -b 6 --amp --lr .036 --sync-bn \
#     --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.99995 \
#     --need-encode-base --no-hard-mining --fpn-name new_pafpn_fa

# add hard-mining liou loss
./distributed_train.sh $GPUS_NUM ../__DATASET/coco --model efficientdet_d2 -b 6 --amp --lr .036 --sync-bn \
    --opt fusedmomentum --warmup-epochs 5 --lr-noise 0.4 0.9 --model-ema --model-ema-decay 0.99995 \
    --need-encode-base --fpn-name new_pafpn_fa
