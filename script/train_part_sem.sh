#!/usr/bin/env bash
# VOC 1-way 1-shot
output_dir="./outputs/PANet"
DATE=`date "+%Y-%m-%d"`
GPU_ID=2
WAY=1
SHOT=2
LR=5e-4
base_loss_scaler=0.1
center=5
share=4
output_sem_size=417
resnet=50

for FOLD in $(seq 0 3)
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
eval=0 \
output_sem_size=$output_sem_size \
center=$center \
base_loss_scaler=$base_loss_scaler \
model.resnet=True \
model.sem=True \
share=$share \
fix=True \
resnet=$resnet \
ckpt_dir="$output_dir/$DATE-voc-$resnet-fix-w$WAY-s$SHOT-lr$LR-cen$center-base$base_loss_scaler-size$output_sem_size-semshare$share-F$FOLD" \
| tee logs/"$DATE-voc-$resnet-fix-w$WAY-s$SHOT-lr$LR-cen$center-base$base_loss_scaler-size$output_sem_size-semshare$share-F$FOLD".txt
done