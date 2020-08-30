#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=0
WAY=1
SHOT=1
LR=5e-4

for FOLD in $(seq 0 3)
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
n_steps=24000 \
eval=0 \
model.resnet=True \
model.sem=False \
ckpt_dir="$output_dir/$DATE-voc-w$WAY-s$SHOT-lr$LR-resnet-F$FOLD" \
| tee logs/"$DATE-voc-w$WAY-s$SHOT-lr$LR-resnet-F$FOLD".txt
done



