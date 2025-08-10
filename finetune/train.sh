export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.py +experiment=mdr-dec6-union
