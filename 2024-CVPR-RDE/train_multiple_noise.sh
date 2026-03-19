#!/bin/bash

# Script để train trên nhiều mức nhiễu khác nhau

root_dir=/mnt/nvme0/home/hoaixg/RDE-mSiglip/2024-CVPR-RDE
tau=0.015 
margin=0.3
select_ratio=0.3
loss=TAL
DATASET_NAME=VN3K-VI

# Danh sách các mức nhiễu muốn train
# Có thể thay đổi danh sách này
NOISY_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

# Tên model chuẩn của SigLIP trên Hugging Face
MODEL_NAME="google/siglip-base-patch16-256-multilingual"

echo "=========================================="
echo "TRAIN TRÊN NHIỀU MỨC NHIỄU"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Mức nhiễu sẽ train: ${NOISY_RATES[@]}"
echo "=========================================="

# Tạo thư mục noiseindex nếu chưa có
mkdir -p ./noiseindex

# Lặp qua từng mức nhiễu
for noisy_rate in "${NOISY_RATES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Bắt đầu train với noisy_rate = $noisy_rate"
    echo "=========================================="
    
    noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
    
    # Kiểm tra xem file noise index có tồn tại không
    if [ ! -f "$noisy_file" ] && [ "$noisy_rate" != "0.0" ]; then
        echo "⚠ Cảnh báo: File $noisy_file không tồn tại!"
        echo "File sẽ được tạo tự động khi training, nhưng có thể mất thời gian."
        echo "Bạn có muốn tiếp tục? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "Bỏ qua noisy_rate = $noisy_rate"
            continue
        fi
    fi
    
    # Chạy training
    CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --noisy_rate $noisy_rate \
        --noisy_file $noisy_file \
        --name RDE \
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
        --pretrain_choice $MODEL_NAME
    
    # Kiểm tra exit code
    if [ $? -eq 0 ]; then
        echo "✓ Hoàn thành training với noisy_rate = $noisy_rate"
    else
        echo "✗ Lỗi khi training với noisy_rate = $noisy_rate"
        echo "Bạn có muốn tiếp tục với mức nhiễu tiếp theo? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "Dừng lại..."
            exit 1
        fi
    fi
    
    echo "Chờ 3 giây trước khi chạy mức nhiễu tiếp theo..."
    sleep 3
done

echo ""
echo "=========================================="
echo "Hoàn thành train tất cả các mức nhiễu!"
echo "=========================================="









