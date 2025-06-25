# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
from math import exp
import sys
import os
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from  util.metric_iou import mean_IU

import numpy as np
from scipy import signal, ndimage
from torch.autograd import Variable
import torch.nn.functional as F

from PIL import Image
import PIL.ImageOps
import torchvision.utils as vutils

def normalize_image_tensor(img_tensor):
    """
    将图像张量的像素值归一化到 0-1 范围。

    Args:
    img_tensor (Tensor): 图像张量。

    Returns:
    Tensor: 归一化后的图像张量。
    """
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float()  # 转换为浮点类型
        img_tensor /= 255.0  # 归一化到 0-1 范围
    return img_tensor

#---------------------------------------------------
# stole from Boqiang Zhang
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size //
                       2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    print('*' * 80)
    print(size, img1.shape,window.shape )
    print('*' * 80)
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(img1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(img2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[:: 2, :: 2]
        im2 = filtered_im2[:: 2, :: 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0: level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0: level - 1]), weight[0: level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power

def get_msssim(gts, imgs_pred):
    gts = gts.squeeze(0)
    imgs_pred = imgs_pred.squeeze(0)
    R = gts[0, :, :]
    G = gts[1, :, :]
    B = gts[2, :, :]
    YGT = .299 * R + .587 * G + .114 * B
    R = imgs_pred[0, :, :]
    G = imgs_pred[1, :, :]
    B = imgs_pred[2, :, :]
    YBC = .299 * R + .587 * G + .114 * B
    mssim = msssim(np.array(YGT * 255), np.array(YBC * 255))
    return mssim

def mse_psnr(gt, image):
    mse = torch.mean((gt - image)**2, dim=(1,2,3))
    mean_mse = torch.mean(mse)
    mean_psnr = torch.mean(10 * torch.log10(1 / (mse+1e-6)))
    return mean_mse, mean_psnr

# stole from Tongkun Guan
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
#--------------------------------------------------

def calculate_iou(preds, targets, threshold=0.5):
    """
    计算预测和目标之间的 IoU，允许自定义阈值。
    
    Args:
    preds (Tensor): 预测的分割图像的批次，形状为 [batch_size, height, width]。
    targets (Tensor): 真实的分割图像的批次，形状为 [batch_size, height, width]。
    threshold (float): 用于二值化预测的阈值。
    
    Returns:
    float: 平均 IoU 分数。

    示例使用：
    preds = ...  # 模型的预测输出
    targets = ...  # 真实的标签
    average_iou = calculate_iou(preds, targets, threshold=0.7)  # 例如，使用0.7作为阈值
    """
    # 应用阈值
    preds = preds > threshold
    targets = targets > 0.5  # 真实标签通常已经是二值的，但如果不是，也可以设置阈值

    # 计算交集和并集
    intersection = torch.logical_and(preds, targets).float().sum((1, 2))  # 沿批次和空间维度求和
    union = torch.logical_or(preds, targets).float().sum((1, 2))

    # 计算 IoU
    iou = intersection / union
    iou[union == 0] = 1  # 如果并集为 0，则 IoU 定义为 1

    return iou.mean()  # 返回平均 IoU 分数


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, wandb_logger=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('res_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('align_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (images, gts) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if args.debug and data_iter_step == 50:
            print("stop training at {}".format(data_iter_step))
            break
        
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, loss_outputs, pred = model(images, gts)
            metric_logger.update(**loss_outputs)

        loss_value = loss.item()
    
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    nums = 0
    sum_mse = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_iou = 0
    for data_iter_step, (images, gts) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        nums += images.shape[0]
        if args.debug and data_iter_step == 50:
            print("stop testing at {}".format(data_iter_step))
            break

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        loss, loss_outputs, pred = model(images, gts)


        if args.pixel_type == 'SR':
            imgs_pred = model.unpatchify(pred) # [N,3,H,W]
            mean_mse, mean_psnr = mse_psnr(gts, imgs_pred)
            sum_mse += mean_mse
            sum_psnr += mean_psnr            
            mssim = get_ssim(imgs_pred, gts)
            sum_ssim += mssim


            if (data_iter_step+1) % 500:
                print(f"batch:{data_iter_step+1} mse:{mean_mse:.4f} psnr:{mean_psnr:.4f} ssim:{mssim:.4f}\
                     avg_mse:{sum_mse.item()/(data_iter_step+1):.4f} avg_psnr:{sum_psnr.item()/(data_iter_step+1):.4f} avg_ssim:{sum_ssim.item()/(data_iter_step+1):.4f}")

            if args.vis:
                visulize_batch(imgs_pred, gts, args, data_iter_step)

        elif args.pixel_type == 'Seg':
            imgs_pred = model.maskunpatchify(pred) # [N,1,H,W]
            mean_iou = calculate_iou(imgs_pred, gts)
            sum_iou += mean_iou
            if (data_iter_step+1) % 500:
                print(f"batch:{data_iter_step+1} IoU:{mean_iou:.4f} Avg Iou:{(sum_iou.item()/(data_iter_step+1)):.4f}")

            if args.vis:
                preds = (imgs_pred > 0.5).int().float()
                targets = (gts > 0.5).int().float()
                # print(preds.shape,targets.shape)
                visulize_batch(preds, targets, args, data_iter_step)
    if args.pixel_type == 'SR':
        print('Eval MSE: ', sum_mse.item()/(data_iter_step+1), 'Eval PSNR: ', sum_psnr.item()/(data_iter_step+1), 'Eval SSIM: ', sum_ssim.item()/(data_iter_step+1))
    elif args.pixel_type == 'Seg':
        print('Eval IoU: ', sum_iou.item()/(data_iter_step+1))

@torch.no_grad()
def train_valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, best_metric=None,
                    args=None):
    model.eval()
    nums = 0
    sum_mse = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_iou = 0
    for data_iter_step, (images, gts) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        nums += images.shape[0]
        if args.debug and data_iter_step == 50:
            print("stop testing at {}".format(data_iter_step))
            break

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        loss, loss_outputs, pred = model(images, gts)

        if args.pixel_type == 'SR':
            imgs_pred = model.module.unpatchify(pred) # [N,3,H,W]
            mean_mse, mean_psnr = mse_psnr(gts, imgs_pred)
            sum_mse += mean_mse
            sum_psnr += mean_psnr            
            mssim = get_ssim(imgs_pred, gts)
            sum_ssim += mssim

            if (data_iter_step+1) % 500:
                print(f"batch:{data_iter_step+1} mse:{mean_mse:.4f} psnr:{mean_psnr:.4f} ssim:{mssim:.4f}\
                     avg_mse:{sum_mse.item()/(data_iter_step+1):.4f} avg_psnr:{sum_psnr.item()/(data_iter_step+1):.4f} avg_ssim:{sum_ssim.item()/(data_iter_step+1):.4f}")
        
        elif args.pixel_type == 'Seg':
            imgs_pred = model.module.maskunpatchify(pred) # [N,1,H,W]
            mean_iou = calculate_iou(imgs_pred, gts)
            sum_iou += mean_iou
            if (data_iter_step+1) % 500:
                print(f"batch:{data_iter_step+1} IoU:{mean_iou:.4f} Avg Iou:{sum_iou.item()/(data_iter_step+1):.4f}")

    if args.pixel_type == 'SR':
        BEST_PSNR = best_metric['BEST_PSNR']
        BEST_SSIM = best_metric['BEST_SSIM']
        AVG_MSE = sum_mse.item()/(data_iter_step+1)
        AVG_PSNR = sum_psnr.item()/(data_iter_step+1)
        AVG_SSIM = sum_ssim.item()/(data_iter_step+1)

        best_metric['CUR_PSNR'] = AVG_PSNR
        best_metric['CUR_SSIM'] = AVG_SSIM


        if AVG_PSNR > BEST_PSNR: 
            best_metric['BEST_PSNR'] = AVG_PSNR
           
        if AVG_SSIM > BEST_SSIM:
            best_metric['BEST_SSIM'] = AVG_SSIM

        print('Old BEST PSNR: ', BEST_PSNR, 'Old BEST SSIM: ', BEST_SSIM) 
        print('Eval MSE: ', AVG_MSE, 'Eval PSNR: ', AVG_PSNR, 'Eval SSIM: ', AVG_SSIM)
        print('BEST PSNR: ', best_metric['BEST_PSNR'], 'BEST SSIM: ', best_metric['BEST_SSIM'])
    
    elif args.pixel_type == 'Seg':
        BEST_IoU = best_metric['BEST_IoU']
        AVG_IoU = sum_iou.item()/(data_iter_step+1)
        
        if AVG_IoU > BEST_IoU: 
            best_metric['BEST_IoU'] = AVG_IoU

        print('Old BEST_IoU: ', BEST_IoU)
        print('Eval IoU: ', AVG_IoU)
        print('BEST_IoU: ', best_metric['BEST_IoU'])

    return best_metric

def visulize_batch(preds, gts, args,idx):
    '''
    preds: [B,C,H,W]
    gts: [B,C,H,W]
    '''
    save_path = args.demo_dir
    num_per_row = gts.shape[0]
    seq_img_list = [gts,preds]
    imgs_set = torch.cat(seq_img_list,0)
    # B C H W 
    vutils.save_image(imgs_set, os.path.join(save_path , str(idx) +'.jpg'), nrow=num_per_row, normalize=True, scale_each=True)

def visulize_batch_compare(preds_scratch,preds_dig,preds_ccd,preds, gts, args,idx):
    '''
    preds: [B,C,H,W]
    gts: [B,C,H,W]
    '''
    save_path = args.demo_dir
    num_per_row = gts.shape[0]
    seq_img_list = [preds_scratch,preds_dig,preds_ccd,preds,gts]
    imgs_set = torch.cat(seq_img_list,0)
    # B C H W 
    vutils.save_image(imgs_set, os.path.join(save_path , str(idx) +'.jpg'), nrow=num_per_row, normalize=True, scale_each=True)


@torch.no_grad()
def valid_one_epoch_compare(model: torch.nn.Module,
                    model_scratch: torch.nn.Module,
                    model_dig: torch.nn.Module,
                    model_ccd: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    nums = 0
    sum_mse = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_iou = 0
    for data_iter_step, (images, gts) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        nums += images.shape[0]
        if args.debug and data_iter_step == 50:
            print("stop testing at {}".format(data_iter_step))
            break

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        loss, loss_outputs, pred = model(images, gts)
        loss, loss_outputs, pred_scratch = model_scratch(images, gts)
        loss, loss_outputs, pred_dig = model_dig(images, gts)
        loss, loss_outputs, pred_ccd = model_ccd(images, gts)

        print(f"batch: {data_iter_step+1}/{len(data_loader)}")
        if args.pixel_type == 'SR':
            imgs_pred = model.unpatchify(pred) # [N,3,H,W]
            imgs_pred_scratch = model_scratch.unpatchify(pred_scratch) # [N,3,H,W]
            imgs_pred_dig = model_dig.unpatchify(pred_dig) # [N,3,H,W]
            imgs_pred_ccd = model_ccd.unpatchify(pred_ccd) # [N,3,H,W]
            if args.vis:
                visulize_batch_compare(imgs_pred_scratch, imgs_pred_dig, imgs_pred_ccd, imgs_pred, gts, args, data_iter_step)

        elif args.pixel_type == 'Seg':
            imgs_pred = model.maskunpatchify(pred) # [N,1,H,W]
            imgs_pred_scratch = model_scratch.maskunpatchify(pred_scratch) # [N,1,H,W]
            imgs_pred_dig = model_dig.maskunpatchify(pred_dig) # [N,1,H,W]
            imgs_pred_ccd = model_ccd.maskunpatchify(pred_ccd) # [N,1,H,W]
            if args.vis:
                preds_scratch = (imgs_pred_scratch > 0.5).int().float()
                preds_dig = (imgs_pred_dig > 0.5).int().float()
                preds_ccd = (imgs_pred_ccd > 0.5).int().float()
                preds = (imgs_pred > 0.5).int().float()
                targets = (gts > 0.5).int().float()
                visulize_batch_compare(preds_scratch, preds_dig, preds_ccd, preds, targets, args, data_iter_step)

 
