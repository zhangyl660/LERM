#!/usr/bin/env bash

# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python bnm.py /data/SSL/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet18 \
  --lr 0.03 --finetune --seed 0 --log logs/cifar10/bnm/4_labels_per_class
  

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python bnm.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.01 --finetune --seed 0 --log logs/cifar100/bnm/4_labels_per_class

CUDA_VISIBLE_DEVICES=0 python bnm.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 1 -a resnet50 \
  --lr 0.01 --finetune --seed 0 --log logs/cifar100/bnm/1_labels_per_class


# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python bnm.py /data/SSL/DTD -d DTD --num-samples-per-class 4 -a resnet50 \
  --lr 0.03 --finetune --seed 0 --log logs/DTD/bnm/1_labels_per_class

CUDA_VISIBLE_DEVICES=0 python bnm.py /data/SSL/DTD -d DTD --num-samples-per-class 1 -a resnet50 \
  --lr 0.03 --finetune --seed 0 --log logs/DTD/bnm/4_labels_per_class

