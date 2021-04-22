#!/bin/bash


DATASETS=(
    'split_face_direction_new'
)

NUM_CLASSES=(
    5
)

INIT_LR=(
    1e-3
)

GPU_ID=0,1,2,3
ARCH='resnet50'
FINETUNE_EPOCHS=100

# ResNet50 pretrained on ImageNet
echo {\"imagenet\": \"0.7616\"} > logs/baseline_imagenet_acc_${ARCH}.txt


CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
    --arch $ARCH \
    --dataset $DATASETS --num_classes $NUM_CLASSES \
    --lr $INIT_LR \
    --weight_decay 4e-5 \
    --save_folder checkpoints/baseline/experiment2/$ARCH/$DATASETS \
    --epochs $FINETUNE_EPOCHS \
    --mode finetune \
    --logfile logs/baseline_imagenet_acc_${ARCH}.txt \
    --use_imagenet_pretrained