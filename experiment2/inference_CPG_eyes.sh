#!/bin/bash


DATASETS=(
    'split_eyes'
)

NUM_CLASSES=(
    2
)

GPU_ID=0
NETWORK_WIDTH_MULTIPLIER=1.0
ARCH='resnet50'
PRED_PATH='/home/CPG/predict_test'


CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
    --arch $ARCH \
    --dataset ${DATASETS} --num_classes ${NUM_CLASSES} \
    --load_folder checkpoints/CPG/experiment2/$ARCH/${DATASETS}/gradual_prune \
    --mode inference \
    --jsonfile logs/baseline_imagenet_acc_$ARCH.txt \
    --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
    --log_path logs/imagenet_inference.log \
    --pred_path $PRED_PATH
