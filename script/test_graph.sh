#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=0
#FOLD=0
WAY=1
SHOT=1
UN=4
LR=5e-4
base_loss_scaler=0.1
center=5
share=4
pt_lambda=0.8
p_value_thres=0.5
resnet=50
topk=30
un_bs=3

for FOLD in $(seq 0 0)
do
python train_graph.py with \
gpu_id=$GPU_ID \
mode='train' \
label_sets=$FOLD \
model.part=True \
task.n_ways=$WAY \
task.n_shots=$SHOT \
task.n_unlabels=$UN \
optim.lr=$LR \
evaluate_interval=4000 \
infer_max_iters=1000 \
num_workers=8 \
n_steps=20000 \
eval=1 \
center=$center \
base_loss_scaler=$base_loss_scaler \
share=$share \
fix=True \
segments=True \
pt_lambda=$pt_lambda \
p_value_thres=$p_value_thres \
resnet=$resnet \
topk=$topk \
un_bs=$un_bs \
ckpt_dir="$output_dir/PANet/2020-06-10-voc-50-graph-w1-s1-un4-lr5e-4-cen5-lam0.8-p0.5-topk30-unbs3-F$FOLD"
done