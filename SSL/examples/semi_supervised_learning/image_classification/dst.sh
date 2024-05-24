#!/usr/bin/env bash
# ImageNet Supervised Pretrain (ResNet50)
# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python dst.py /data/SSL/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/cifar10/dst/4_labels_per_class

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python dst.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/cifar100/dst/4_labels_per_class

CUDA_VISIBLE_DEVICES=0 python dst.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 1 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.7 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/cifar100/dst/1_labels_per_class

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python dst.py /data/SSL/DTD -d DTD --num-samples-per-class 4 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.95 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/DTD/dst/4_labels_per_class

CUDA_VISIBLE_DEVICES=0 python dst.py /data/SSL/DTD -d DTD --num-samples-per-class 1 -a resnet50 \
  --lr 0.003 --finetune --threshold 0.95 --trade-off-self-training 1 --eta-prime 2 \
  --seed 0 --log logs/DTD/dst/1_labels_per_class
