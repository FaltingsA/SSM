CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 512 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 2  \
--output_dir /home/gaoz/output/flip/base-flip-sb2048-debug \
--data_path /home/gaoz/datasets/real-rec/


CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 1024 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 2  \
--output_dir /root/autodl-tmp/output/base-flip/base-hflip-2048 \
--data_path /root/autodl-tmp/unidata/real-rec/
--direction HF --num_workers 10


CUDA_VISIBLE_DEVICES=0,1 python3 -m tor
ch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 1024 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 2  \
--output_dir /root/autodl-tmp/output/base-flip/base-hflip-2048 \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction HF --num_workers 10


CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29050 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 1024 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 2  \
--output_dir /root/autodl-tmp/output/base-flip/base-vflip-2048 \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction VF --num_workers 10


CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 512 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 1  \
--output_dir /root/autodl-tmp/output/base-flip/trans-judgue-1024 \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction Hybrid --num_workers 10


CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 384s \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 1  \
--output_dir /root/autodl-tmp/output/base-flip/trans-hybird-mul-prob \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction Hybrid --num_workers 10


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 256 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 1  \
--output_dir /root/autodl-tmp/output/base-flip/trans-hybird-mulemb3-prob \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction Hybrid --num_workers 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 256 \
--mode single \
--model flipae_vit_small_str   \
--epochs 20 \
--warmup_epochs 1  \
--output_dir /root/autodl-tmp/output/base-flip/trans-hybird-mulemb3-prob-p4x8 \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction Hybrid --num_workers 10


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29055 flip_pretrain.py  \
--lr 1e-3 \
--batch_size 256 \
--mode single \
--model flipae_vit_small_strdim384 \
--epochs 20 \
--warmup_epochs 1  \
--output_dir /root/autodl-tmp/output/base-flip/emb3-prob4x4-dim384 \
--data_path /root/autodl-tmp/unidata/real-rec/ \
--direction Hybrid --num_workers 10