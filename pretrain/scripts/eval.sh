CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29058 demo.py  \
--batch_size 10 \
--model flipae_vit_small_str \
--model_path /home/gaoz/pretrained/trans-hybird-mulemb3-prob/checkpoint-19.pth \
--data_path /home/gaoz/datasets/data/test/data_shuffle/ \
--demo_dir /home/gaoz/output/HF-shuffle-nomask/ \
--direction HF --num_workers 10 \
--debug


CUDA_VISIBLE_DEVICES=6 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29058 demo.py  \
--batch_size 10 \
--model flipae_vit_small_str \
--model_path /home/gaoz/output/mdr-HF-pair-mix/checkpoint-9.pth \
--data_path /home/gaoz/datasets/data/test/IIIT5k/ \
--demo_dir /home/gaoz/output/pair-hf-mix-9/IIIT5k/ \
--direction HF --num_workers 10 --pair_mix


SVTP SVT salient artistic IC13_1095 multi_words IC15_2077

CUDA_VISIBLE_DEVICES=4 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29058 demo.py  \
--batch_size 10 \
--model flipae_vit_small_str \
--model_path /home/gaoz/output/mdr-union-pool-b256/checkpoint-19.pth \
--data_path /home/gaoz/datasets/test/IIIT5k/ \
--demo_dir /home/gaoz/output/vis/IIIT5k/RO \
--direction RO --num_workers 10 \
--debug
