CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port 29060 main_pixel.py  \
--batch_size 1 \
--model pixel_vit_small \
--pixel_type Seg \
--demo_dir /home/gaoz/output_pixel/Seg-100e-norm/best_vis/ \
--eval_data_path /home/gaoz/datasets/TextSeg/TextSeg/image_slice/test/ \
--model_path /home/gaoz/output_pixel/Seg-100e-norm/best/IoU-0.8288-checkpoint-39.pth \
--num_workers 10 --eval_only --vis
