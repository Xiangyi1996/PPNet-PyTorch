#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=1
WAY=2
SHOT=5
LR=5e-4

for FOLD in $(seq 0 1)
do
python train_fewshot.py with \
gpu_id=$GPU_ID \
mode='train' \
label_sets=$FOLD \
model.part=False \
task.n_ways=$WAY \
task.n_shots=$SHOT \
optim.lr=$LR \
evaluate_interval=4000 \
infer_max_iters=1000 \
num_workers=8 \
n_steps=20000 \
eval=1 \
vis_pred=0 \
model.resnet=True \
resnet_pretrain=False \
model.sem=False \
ckpt_dir="$output_dir/2020-02-15-voc-w1-s1-lr5e-4-resnet-debug-F$FOLD"
done



