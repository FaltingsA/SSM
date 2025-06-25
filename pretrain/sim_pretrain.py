# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from torch.utils.data import Dataset, ConcatDataset, Subset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_sim_re


# from PreNet import DualAutoencoderViT
# from engine_pretrain import train_one_epoch, LmdbDataset, AlignCollate, hierarchical_dataset

from engine_pretrain import train_one_epoch, train_one_epoch_test
from datasets import LmdbDataset, AlignCollate, AlignCollate_test, hierarchical_dataset




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mode', default='align', type=str,
                        help='mode [align,single]')

    parser.add_argument('--model', default='flipae_vit_small_str', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=[32,128], type=list,
                        help='images input size')
    parser.add_argument('--patch_size', default=[4,4], type=list,
                        help='images input size')

    parser.add_argument('--embed_dim', default=384, type=int,
                        help='encoder embedding dim')

    parser.add_argument('--encoder_depth', default=12, type=int,
                        help='encoder depth')

    parser.add_argument('--encoder_num_heads', default=6, type=int,
                        help='encoder heads num')

    parser.add_argument('--align_encoder', default='clip_ViT-B/16', type=str,
                        help='choose the pretrained align_encoder checkpoint,\
                        [RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px]')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--online_ln', default=False, action='store_true', help='also use frozen LN in online branch')
    parser.add_argument('--with_blockwise_mask', default=False, action='store_true')
    parser.add_argument('--blockwise_num_masking_patches', default=75, type=int)

    # hyper-parameter
    parser.add_argument('--mm', default=0.995, type=float)
    parser.add_argument('--mmschedule', default='const')
    parser.add_argument('--lambda_F', default=50, type=float) # may no need
    parser.add_argument('--T', default=0.2, type=float)       # check
    parser.add_argument('--clip_grad', default=1.0, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/gaoz/dataset/unidata/data/val/', type=str,
                        help='dataset path')
    parser.add_argument('--select_data', type=str, default='label-unlabel', # label-unlabel
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--demo_dir', default='./vis_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--rgb', action='store_true', default=True, help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--direction', default='Hybrid', type=str,
                        help='direction of Flip')
    parser.add_argument('--ratio', default=1, type=int, help='p of Flip')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # selective augmentation 
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', default=False, help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None, help='Magnitude of data augment groups to apply. None if random.')

    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='') 
    parser.add_argument('--demo_imgs', default='')

    parser.add_argument('--debug', action='store_true', help='Use debug mode ')
    # pair_mix
    parser.add_argument('--pair_mix', action='store_true', help='Use debug mode ')

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', action='store_true', 
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='mdr-debug', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    # parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
    #                     help="Save model checkpoints as W&B Artifacts.")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    args.eval = False
    # # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train, _ = hierarchical_dataset(root=args.data_path, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = misc.WandbLogger(args)
    else:
        wandb_logger = None

    AlignCollate_ = AlignCollate_test(imgH=args.input_size[0], imgW=args.input_size[1], keep_ratio_with_pad=args.PAD, opt=args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=AlignCollate_,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_sim_re.__dict__[args.model](img_size=args.input_size, \
                                patch_size=args.patch_size, \
                                norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if wandb_logger:
            wandb_logger.set_steps()

        train_stats = train_one_epoch_test(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=None, wandb_logger=wandb_logger,
            args=args
        )

        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.demo_dir:
        Path(args.demo_dir).mkdir(parents=True, exist_ok=True)
    main(args)
