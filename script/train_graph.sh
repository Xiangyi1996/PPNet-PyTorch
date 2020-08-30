#!/usr/bin/env bash
# Experiment Parameters
# VOC 1/2-way   |  1-shot |  5-shot
# pt_lambda     |   0.8   |    0.9
# p_value_thres |   0.5   |    0.9
# topk          |    30   |      5
# un_bs         |     2   |      6

output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=0
WAY=1
SHOT=1
UN=6
LR=5e-4
base_loss_scaler=0.1
center=5
share=4
pt_lambda=0.8
p_value_thres=0.5
resnet=50
topk=30
un_bs=2


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
n_steps=24000 \
eval=0 \
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
ckpt_dir="$output_dir/$DATE-voc-$resnet-graph-w$WAY-s$SHOT-un$UN-lr$LR-cen$center-lam$pt_lambda-p$p_value_thres-topk$topk-unbs$un_bs-F$FOLD" \
| tee logs/"$DATE-voc-$resnet-graph-w$WAY-s$SHOT-un$UN-lr$LR-cen$center-lam$pt_lambda-p$p_value_thres-topk$topk-unbs$un_bs-F$FOLD".txt
done