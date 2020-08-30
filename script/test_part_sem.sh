#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=0
#FOLD=1
WAY=2
SHOT=5
LR=5e-4
base_loss_scaler=0.1
center=5
share=4
aspp='old'
output_sem_size=417
resnet=50

for FOLD in $(seq 0 0)
do
python train_part_sem.py with \
gpu_id=$GPU_ID \
mode='train' \
label_sets=$FOLD \
model.part=True \
task.n_ways=$WAY \
task.n_shots=$SHOT \
optim.lr=$LR \
evaluate_interval=4000 \
infer_max_iters=1000 \
num_workers=8 \
n_steps=24000 \
eval=1 \
output_sem_size=$output_sem_size \
center=$center \
base_loss_scaler=$base_loss_scaler \
model.resnet=True \
resnet_pretrain=False \
model.sem=True \
polynomialLR=False \
share=$share \
aspp=$aspp \
fix=True \
resnet=$resnet \
ckpt_dir="$output_dir/2020-02-26-voc-graph-w2-s1-un6-lr5e-4-cen5-lam0.8-p0.5-F$FOLD"
done