#!/bin/bash

# 定义第一个数组
first_array=(IIIT5k SVTP SVT salient artistic IC13_1095 multi_words IC15_2077)

# 定义第二个数组
second_array=(HF VF RO)

# 第一层循环遍历第一个数组
for i in "${first_array[@]}"; do
  
  # 第二层循环遍历第二个数组
  for j in "${second_array[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29058 demo.py  \
    --batch_size 10 \
    --model flipae_vit_small_str \
    --model_path /home/gaoz/output/mdr-union-pool-b256/checkpoint-19.pth \
    --data_path /home/gaoz/datasets/test/${i}/ \
    --demo_dir /home/gaoz/output/vis/${i}/${j} \
    --direction ${j} --num_workers 10 
  done
  
done
