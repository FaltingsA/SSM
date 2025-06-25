CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29060 main_pixel.py  \
--batch_size 10 \
--model pixel_vit_small \
--pixel_type SR \
--demo_dir /home/gaoz/output_pixel/SR-100e-noaug_norm/hard_best_vis/ \
--eval_data_path /home/gaoz/datasets/textzoom/test/hard \
--model_path /home/gaoz/output_pixel/SR-100e-noaug_norm/best/psnr-22.1962-ssim-0.7610-checkpoint-72.pth \
--num_workers 10 --eval_only --vis