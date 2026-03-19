#!/usr/bin/env python3
"""
Script kiểm tra các file noise index đã có
"""
import os
import os.path as op
import numpy as np

def check_noise_files(dataset_name="VN3K-VI", root_dir="/mnt/nvme0/home/hoaixg/RDE-mSiglip/2024-CVPR-RDE"):
    """Kiểm tra các file noise index đã có"""
    noiseindex_dir = op.join(root_dir, 'noiseindex')
    
    print("="*60)
    print(f"KIỂM TRA FILE NOISE INDEX CHO {dataset_name}")
    print("="*60)
    
    # Các mức nhiễu thường dùng
    noisy_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    existing_files = []
    missing_files = []
    
    for noisy_rate in noisy_rates:
        noisy_file = op.join(noiseindex_dir, f'{dataset_name}_{noisy_rate}.npy')
        if op.exists(noisy_file):
            file_size = os.path.getsize(noisy_file)
            # Kiểm tra file có hợp lệ không
            try:
                data = np.load(noisy_file)
                existing_files.append((noisy_rate, noisy_file, file_size, len(data)))
            except:
                print(f"⚠ File {noisy_file} tồn tại nhưng có thể bị hỏng!")
                missing_files.append(noisy_rate)
        else:
            missing_files.append(noisy_rate)
    
    print(f"\n✓ Các file đã có ({len(existing_files)}):")
    for noisy_rate, file_path, size, length in existing_files:
        size_mb = size / (1024 * 1024)
        print(f"  - noisy_rate={noisy_rate:.1f}: {file_path}")
        print(f"    Size: {size_mb:.2f} MB, Length: {length}")
    
    if missing_files:
        print(f"\n✗ Các file chưa có ({len(missing_files)}):")
        for noisy_rate in missing_files:
            print(f"  - noisy_rate={noisy_rate:.1f}: {op.join(noiseindex_dir, f'{dataset_name}_{noisy_rate}.npy')}")
        print("\nLưu ý: File sẽ được tạo tự động khi training nếu chưa có.")
    else:
        print("\n✓ Tất cả các file noise index đã có sẵn!")
    
    print("="*60)
    return existing_files, missing_files

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="VN3K-VI")
    parser.add_argument("--root_dir", type=str, 
                        default="/mnt/nvme0/home/hoaixg/RDE-mSiglip/2024-CVPR-RDE")
    args = parser.parse_args()
    
    check_noise_files(args.dataset_name, args.root_dir)









