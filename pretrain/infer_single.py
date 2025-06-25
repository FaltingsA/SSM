# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from torch.utils.data import Dataset, ConcatDataset, Subset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_re


from engine_pretrain import train_one_epoch, valid_one_epoch, LmdbDataset, AlignCollate, hierarchical_dataset,visualize_pair,mixup_filp_data




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--show_iter', default=20, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='flipae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=[32,128], type=list,
                        help='images input size')

    parser.add_argument('--patch_size', default=[4,4], type=list,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)



    # Dataset parameters
    parser.add_argument('--model_path', default='/home/gaoz/output/flip/flipae_vit_base_patch16_dec256d4b-20em3/checkpoint-19.pth')
    parser.add_argument('--data_path', default='/home/gaoz/datasets/unidata/benckmark/evaluation/', type=str,
                        help='dataset path')
    parser.add_argument('--img_path', default='/home/gaoz/ssl-str/cases/', type=str,
                        help='dataset path')
    parser.add_argument('--demo_dir', default='/home/gaoz/output/flip/demo/20em3-19/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

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
    parser.add_argument('--direction', default='HF', type=str,
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

    # # for comparison to other augmentations
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

    return parser

def infer_batch(model: torch.nn.Module, samples_origin:torch.tensor,
                    device: torch.device, 
                    log_writer=None,
                    args=None):
    model.eval()
    accum_iter = args.show_iter
    loss_show = 0

    

    samples_new = copy.deepcopy(samples_origin)
    images_mixed_list = []
    images_flipped_list = []
    pos_list = []
    neg_list = []
    
    image_mixed0, image_flipped0, pos, neg = mixup_filp_data(samples_origin[0], samples_new[0],alpha=0.5,direction=args.direction,ratio=args.ratio)
    for image_o, image_n in zip(samples_origin, samples_new):
        image_mixed, image_flipped, pos, neg = mixup_filp_data(image_o, image_n,alpha=0.5,direction=args.direction,ratio=args.ratio)
        
        ### add some mask noise 
        # noise_mask = torch.ones_like(image_mixed)
        # # c,h,w = image_mixed.shape
        # noise_mask[:, :, 20:38] = 0
        # noise_mask[:, :, 90:108] = 0
        # noise_mask[:, :, :64] = 0
        # image_mixed_mask = image_mixed * noise_mask
        # images_mixed_list.append(image_mixed_mask)

        ### add mix 
        # image_mixed[:, :, :32] = image_mixed0[:, :, :32]
        # image_mixed[:, :, 96:] = image_mixed0[:, :, 96:]

        ### add partial mix 
        # image_mixed[:, :, :] = image_flipped[:, :, 80:]
        # image_mixed[:, :, 64:] = image_flipped[:, :, 64:]

        images_mixed_list.append(image_mixed)
        images_flipped_list.append(image_flipped)
        pos_list.append(pos)
        neg_list.append(neg)

    samples = torch.cat([t.unsqueeze(0) for t in images_mixed_list], 0)
    samples = samples.to(device, non_blocking=True)
    
    samples_flipped = torch.cat([t.unsqueeze(0) for t in images_flipped_list], 0)
    samples_flipped = samples_flipped.to(device, non_blocking=True)
    
    pos_prompt = torch.tensor(pos_list, dtype=torch.long).to(device, non_blocking=True)
    neg_prompt = torch.tensor(neg_list, dtype=torch.long).to(device, non_blocking=True)
    
    samples_origin = samples_origin.to(device, non_blocking=True)
    samples_gt = {}
    samples_gt.update({'pos':samples_origin})
    samples_gt.update({'neg':samples_flipped})

    loss, pred_pos, pred_neg = model(samples, samples_gt, pos_prompt, neg_prompt)

    imgs_pred_pos = model.unpatchify(pred_pos) # [N,3,H,W]
    imgs_pred_neg = model.unpatchify(pred_neg) # [N,3,H,W]

    visualize_pair(imgs_pred_pos,imgs_pred_neg,samples,samples_origin,samples_flipped,args.demo_dir,0)

    loss_value = loss.item()
    loss_show += loss_value
    print("avg_loss:{0}".format(loss/accum_iter))
       

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
    args.eval = True
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_eval, _ = hierarchical_dataset(root=args.data_path, args=args)
    print(dataset_eval)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_eval = %s" % str(sampler_eval))
    else:
        sampler_eval = torch.utils.data.RandomSampler(dataset_eval)

   
    log_writer = None

    # AlignCollate_ = AlignCollate(imgH=args.input_size[0], imgW=args.input_size[1], keep_ratio_with_pad=args.PAD, opt=args)
    # data_loader_eval = torch.utils.data.DataLoader(
    #     dataset_eval, sampler=sampler_eval,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     collate_fn=AlignCollate_,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    
    # define the model
    model = models_mae_re.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size , norm_pix_loss=args.norm_pix_loss)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    # model_without_ddp = model.module
    
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    loss_scaler = None
    
   
    print(model)
    start_time = time.time()

    img_list = [args.img_path+im for im in os.listdir(args.img_path)]
    img_tensor_list = []
    for img_path in img_list:
        img = Image.open(img_path).convert('RGB')  # for color image
        img = transforms.ToTensor()(img)
        img_tensor_list.append(img)

    img_tensors = torch.stack(img_tensor_list,dim=0)
    infer_batch(model, img_tensors,
        device, log_writer=None, 
        args=args
    )
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testining time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.demo_dir:
        Path(args.demo_dir).mkdir(parents=True, exist_ok=True)
    main(args)
