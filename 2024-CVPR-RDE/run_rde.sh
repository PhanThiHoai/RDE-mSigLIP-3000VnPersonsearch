
root_dir=/mnt/nvme0/home/hoaixg/RDE-mSiglip/2024-CVPR-RDE
tau=0.015 
margin=0.3
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=VN3K-V2E
# CUHK-PEDES ICFG-PEDES RSTPReid VN3K-V2E

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy

# Tên model chuẩn của SigLIP trên Hugging Face
MODEL_NAME="google/siglip-base-patch16-256-multilingual"

CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name mSiglip-RDE \
    --img_aug \
    --txt_aug \
    --batch_size 32 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 60 \
    --pretrain_choice $MODEL_NAME \
    --resume \
    --resume_ckpt_file /mnt/nvme0/home/hoaixg/RDE-mSiglip/2024-CVPR-RDE/run_logs/VN3K-V2E/20260318_141326_mSiglip-RDE_TAL+sr0.3_tau0.015_margin0.3_n0.0/best.pth