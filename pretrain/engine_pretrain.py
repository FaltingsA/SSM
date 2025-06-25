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
sys.path.insert(1, '/data/pylib')
import math
import sys
from typing import Iterable
import copy
import torch
import glob
from pathlib import Path, PurePath
# import logging
import random
import util.misc as misc
import util.lr_sched as lr_sched
import os
import sys
import re
import six
import math
import lmdb
import torch
from typing import Callable, Optional, Union

from augmentation.warp import Curve, Distort, Stretch
from augmentation.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from augmentation.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from augmentation.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from augmentation.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from augmentation.camera import Contrast, Brightness, JpegCompression, Pixelate
from augmentation.weather import Fog, Frost, Rain, Shadow #Snow,
from augmentation.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from natsort import natsorted
from PIL import Image
import PIL.ImageOps
import numpy as np
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = torch.clamp(image,min=0,max=1)
    image = transforms.ToPILImage()(image)
    # print(image.shape)
    return image

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            train_sampler = DistributedSampler(_dataset)
            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                sampler=train_sampler,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text, _ = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text, _ = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts



def hierarchical_dataset(root, args):
    # try:
    #     kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    # except KeyError:
    #     pass
    root = Path(root).absolute()
    dataset_log = f'dataset root:\t{root}\n'
    print(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        print(ds_name,ds_root)
        dataset = LmdbDataset(ds_root, args)
        # dataset_log += f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}\n'
        print(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets), dataset_log




class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    try:
                        label = txn.get(label_key).decode('utf-8') 
                    except AttributeError:
                        label ='text'
                    if len(label) > self.opt.max_length:
                        continue
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)
                
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            try:
                label = txn.get(label_key).decode('utf-8') 
            except AttributeError:
                label ='text'
                
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


def isless(prob=0.5):
    return np.random.uniform(0,1) < prob

class DataAugment(object):
    '''
    Supports with and without data augmentation 
    '''
    def __init__(self, opt):
        self.opt = opt

        if not opt.eval:
            self.process = [Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]
            self.camera = [Contrast(), Brightness(), JpegCompression(), Pixelate()]

            self.pattern = [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]

            self.noise = [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]
            self.blur = [GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]
            self.weather = [Fog(), Frost(), Rain(), Shadow()]

            self.noises = [self.blur, self.noise, self.weather]
            self.processes = [self.camera, self.process]

            self.warp = [Curve(), Distort(), Stretch()]
            self.geometry = [Rotate(), Perspective(), Shrink()]

            self.isbaseline_aug = False
            # rand augment
            if self.opt.isrand_aug:
                self.augs = [self.process, self.camera, self.noise, self.blur, self.weather, self.pattern, self.warp, self.geometry]
            # semantic augment
            elif self.opt.issemantic_aug:
                self.geometry = [Rotate(), Perspective(), Shrink()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.augs = [self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # pp-ocr augment
            elif self.opt.islearning_aug:
                self.geometry = [Rotate(), Perspective()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # scatter augment
            elif self.opt.isscatter_aug:
                self.geometry = [Shrink()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.geometry]
                self.baseline_aug = True
            # rotation augment
            elif self.opt.isrotation_aug:
                self.geometry = [Rotate()]
                self.augs = [self.geometry]
                self.isbaseline_aug = True

        self.scale = False

    def __call__(self, img):
        '''
            Must call img.copy() if pattern, Rain or Shadow is used
        '''
        if self.opt.eval or isless(self.opt.intact_prob):
            pass
        elif self.opt.isrand_aug or self.isbaseline_aug:
            img = self.rand_aug(img)
        # individual augment can also be selected
        elif self.opt.issel_aug:
            img = self.sel_aug(img)

        img = transforms.ToTensor()(img)
        return img

    def rand_aug(self, img):
        augs = np.random.choice(self.augs, self.opt.augs_num, replace=False)
        for aug in augs:
            index = np.random.randint(0, len(aug))
            op = aug[index]
            mag = np.random.randint(0, 3) if self.opt.augs_mag is None else self.opt.augs_mag
            if type(op).__name__ == "Rain"  or type(op).__name__ == "Grid":
                img = op(img.copy(), mag=mag)
            else:
                img = op(img, mag=mag)

        return img

    def sel_aug(self, img):

        prob = 1.
        if self.opt.process:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.process))
            op = self.process[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.noise:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.noise))
            op = self.noise[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.blur:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.blur))
            op = self.blur[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.weather:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.weather))
            op = self.weather[index]
            if type(op).__name__ == "Rain": #or "Grid" in type(op).__name__ :
                img = op(img.copy(), mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        if self.opt.camera:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.camera))
            op = self.camera[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.pattern:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.pattern))
            op = self.pattern[index]
            img = op(img.copy(), mag=mag, prob=prob)

        iscurve = False
        if self.opt.warp:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.warp))
            op = self.warp[index]
            if type(op).__name__ == "Curve":
                iscurve = True
            img = op(img, mag=mag, prob=prob)

        if self.opt.geometry:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.geometry))
            op = self.geometry[index]
            if type(op).__name__ == "Rotate":
                img = op(img, iscurve=iscurve, mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        return img

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        return Pad_img

class AlignCollate_(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt
        

    def get_aug_op(self):
        direction=self.opt.direction
        Hflip = transforms.RandomHorizontalFlip(p=1)
        Vflip = transforms.RandomVerticalFlip(p=1)
        Srotate = transforms.RandomRotation((180,180),expand=True)

        if direction=='HF':
            pos = 0
            neg = 1
            aug_op = Hflip

        elif direction=='VF':
            pos = 0
            neg = 2
            aug_op = Vflip

        elif direction=='Hybrid':
            prob = random.random()
            if prob > 0.5:
                pos = 0
                neg = 1
                aug_op = Hflip
            else:
                pos = 0
                neg = 2
                aug_op = Vflip

        elif direction == 'aug_pool':
            prob = random.random()
            if prob > 0.66:
                pos = 0
                neg = 1
                aug_op = Hflip
            elif prob > 0.33:
                pos = 0
                neg = 2
                aug_op = Vflip
            else:
                pos = 0
                neg = 3
                aug_op = Srotate

        return aug_op, pos, neg

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        if self.opt.eval_img:
            images, labels, img_paths = zip(*batch)
        else:
            images, labels = zip(*batch)
            img_paths = None

        pil = transforms.ToPILImage()
        aug_transform = DataAugment(self.opt)
        
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                aug_images = aug_transform(resized_image)
                pad_images = transform(aug_images)
                resized_images.append(pad_images)
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            # image_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images]
            samples_ori = []
            samples_fli = []
            samples_mix = []
            pos_prompt = []
            neg_prompt = []
            if self.opt.pair_mix:
                for idx in range(0, len(images), 2):
                    ct = aug_transform(images[idx].resize((self.imgW, self.imgH), Image.BICUBIC))
                    nt = aug_transform(images[idx+1].resize((self.imgW, self.imgH), Image.BICUBIC))
    
                    # get paired ori 
                    ori1 = torch.cat([ct[:,:,:64],nt[:,:,64:]],dim=2)
                    ori2 = torch.cat([nt[:,:,:64],ct[:,:,64:]],dim=2)
                    samples_ori.append(ori1)
                    samples_ori.append(ori2)
                    # get aug_op & prompt
                    aug_op, pos, neg = self.get_aug_op()
                    ctf = aug_op(ct)
                    ntf = aug_op(nt)
                    fli1 = torch.cat([ctf[:,:,:64], ntf[:,:,64:]],dim=2)
                    fli2 = torch.cat([ntf[:,:,:64], ctf[:,:,64:]],dim=2)
                    samples_fli.append(fli1)
                    samples_fli.append(fli2)
                    pos_prompt.append(pos)
                    pos_prompt.append(pos)
                    neg_prompt.append(neg)
                    neg_prompt.append(neg)
                    # get mixed 
                    mix1 = 0.5*ori1 + 0.5*fli1
                    mix2 = 0.5*ori2 + 0.5*fli2
                    samples_mix.append(mix1)
                    samples_mix.append(mix2)
            else:
                # aug_op, pos, neg = self.get_aug_op()
                for idx in range(len(images)):
                    ct = aug_transform(images[idx].resize((self.imgW, self.imgH), Image.BICUBIC))
                    # get paired ori 
                    samples_ori.append(ct)
                    # get aug_op & prompt
                    aug_op, pos, neg = self.get_aug_op()
                    ctf = aug_op(ct)
                    samples_fli.append(ctf)
                    pos_prompt.append(pos)
                    neg_prompt.append(neg)
                    # get mixed 
                    mix = 0.5*ct + 0.5*ctf
                    samples_mix.append(mix)

            samples_ori = torch.cat([t.unsqueeze(0) for t in samples_ori], 0)
            samples_fli = torch.cat([t.unsqueeze(0) for t in samples_fli], 0)
            samples_mix = torch.cat([t.unsqueeze(0) for t in samples_mix], 0)
            pos_prompt = torch.tensor(pos_prompt, dtype=torch.long)
            neg_prompt = torch.tensor(neg_prompt, dtype=torch.long)
            # image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return samples_ori, samples_fli, samples_mix, pos_prompt, neg_prompt

class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        if self.opt.eval_img:
            images, labels, img_paths = zip(*batch)
        else:
            images, labels = zip(*batch)
            img_paths = None

        pil = transforms.ToPILImage()
        aug_transform = DataAugment(self.opt)
        
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                aug_images = aug_transform(resized_image)
                pad_images = transform(aug_images)
                resized_images.append(pad_images)
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            image_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, img_paths

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

    def __init__(self, root, opt):

        with open(root) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.data_list = [x.strip() for x in content] 

        self.leng_index = [0] * 26
        self.leng_index.append(0)
        text_length = 1
        for i, line in enumerate(self.data_list):
            label_text = line.split(' ')[0]
            if i > 0 and len(label_text) != text_length:
                self.leng_index[text_length] = i
            text_length = len(label_text)
        
        self.nSamples = len(self.data_list)
        self.batch_size = opt.batch_size
        self.opt = opt

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        sample_ = self.data_list[index].split(' ')
        label = sample_[0]
        img_path = sample_[1]

        try:
            if self.opt.rgb:
                img = Image.open(img_path).convert('RGB')  # for color image
            else:
                img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted read image for ', img_path)
            randi = random.randint(0, self.nSamples-1)
            return self.__getitem__(randi)

        if not self.opt.sensitive:
            label = label.lower() 

        return img, label, img_path


def mixup_filp_data(img_o, alpha=0, direction='HF', ratio=1):
    Hflip = transforms.RandomHorizontalFlip(p=ratio)
    Vflip = transforms.RandomVerticalFlip(p=ratio)
    Srotate = transforms.RandomRotation((180,180),expand=True)
    if direction=='HF':
        img_hf = Hflip(img_o)
        img_flipped = img_hf
        img_mixed = 0.5 * (img_o + img_flipped)
        pos = 0
        neg = 1

    elif direction=='VF':
        img_vf = Vflip(img_o)
        img_flipped = img_vf
        img_mixed = 0.5 * (img_o + img_flipped)
        pos = 0
        neg = 2

    elif direction=='Hybrid':
        prob = random.random()
        if prob > 0.5:
            img_flipped = Hflip(img_o)
            pos = 0
            neg = 1
        else:
            img_flipped = Vflip(img_o)
            pos = 0
            neg = 2
        img_mixed = 0.5 * (img_flipped + img_o)

    elif direction=='RO':
        pos = 0
        neg = 3
        img_flipped = Srotate(img_o)
        img_mixed = 0.5 * (img_o + img_flipped)

    elif direction == 'aug_pool':
        prob = random.random()
        if prob > 0.66:
            pos = 0
            neg = 1
            img_flipped = Hflip(img_o)
        elif prob > 0.33:
            pos = 0
            neg = 2
            img_flipped = Vflip(img_o)
        else:
            pos = 0
            neg = 3
            img_flipped = Srotate(img_o)
        img_mixed = 0.5 * (img_flipped + img_o)

    return img_mixed, img_flipped, pos, neg

def mixup_filp_data2(cur_img_o, next_img_o, alpha=0, direction='HF', ratio=1):
    if direction=='HF':
        img_flipped1 = Hflip(cur_img_o)
        img_flipped2 = Hflip(next_img_o)
        img_mixed1 = alpha * cur_img_o + (1-alpha) * img_flipped1
        img_mixed2 = alpha * next_img_o + (1-alpha) * img_flipped2

        pos = 0
        neg = 1

    #TODO: automix
    # elif direction=='VF':
    #     img_vf = Vflip(img_n)
    #     img_flipped = img_vf
    #     img_mixed = alpha * img_o + (1-alpha) * img_flipped
    #     pos = 0
    #     neg = 2
    #TODO: automix
    # elif direction=='Hybrid':
    #     img_hf = Hflip(img_n)
    #     img_vf = Vflip(img_n)
        
    #     prob = random.random()
    #     if prob > 0.5:
    #         img_flipped = img_hf
    #         pos = 0
    #         neg = 1
    #     else:
    #         img_flipped = img_vf
    #         pos = 0
    #         neg = 2
    #     img_mixed = 0.5 * (img_flipped + img_o)

    return img_mixed1, img_flipped1, img_mixed2, img_flipped2, pos, neg

def visualize(imgs_pred,imgs_mixup,imgs_origin,save_path,idx):
    COL = 1 #指定拼接图片的列数
    ROW = 3 #指定拼接图片的行数
    UNIT_HEIGHT_SIZE = 32 #图片高度
    UNIT_WIDTH_SIZE = 128 #图片宽度
    SAVE_QUALITY = 50 #保存的图片的质量 可选0-100

    Image_origin = tensor_to_PIL(imgs_origin)
    Image_mixup = tensor_to_PIL(imgs_mixup)
    Image_pred = tensor_to_PIL(imgs_pred)
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    
    target.paste(Image_origin, (0 , 0))
    target.paste(Image_mixup, (0 , UNIT_HEIGHT_SIZE))
    target.paste(Image_pred, (0 , 2 * UNIT_HEIGHT_SIZE))
    # target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(os.path.join(save_path , str(idx) +'.jpg'), quality=SAVE_QUALITY) #成品图保存

def visualize_pair(imgs_pred_pos,imgs_pred_neg,imgs_mixup,imgs_gt_pos, imgs_gt_neg, save_path,idx):
    num_per_row = imgs_pred_pos.shape[0]
    seq_img_list = [imgs_mixup,imgs_gt_pos,imgs_pred_pos,imgs_gt_neg,imgs_pred_neg]
    imgs_set = torch.cat(seq_img_list,0)
    # B C H W 
    vutils.save_image(imgs_set, os.path.join(save_path , str(idx) +'.jpg'), nrow=num_per_row, normalize=True, scale_each=True)

def visualize_pair2(samples_ori,samples_flipped, samples, save_path,idx):
    num_per_row = samples_ori.shape[0]

    seq_img_list = [samples_ori,samples_flipped, samples]
    imgs_set = torch.cat(seq_img_list,0)
    # B C H W 
    vutils.save_image(imgs_set, os.path.join(save_path , str(idx) +'.jpg'), nrow=num_per_row, normalize=True, scale_each=True)

# def valid_one_epoch(model: torch.nn.Module,
#                     data_loader: Iterable, 
#                     device: torch.device, epoch: int, loss_scaler,
#                     log_writer=None,
#                     args=None):
#     model.eval()
#     accum_iter = args.show_iter
#     loss_show = 0
#     nums = 0
#     for data_iter_step, (samples_origin,_, _) in enumerate(data_loader):
#         # we use a per iteration (instead of per epoch) lr scheduler
#         nums += samples_origin.shape[0]
#         if args.debug and data_iter_step == 50:
#             print("stop testing at {}".format(data_iter_step))
#             break

#         samples_new = copy.deepcopy(samples_origin)
#         images_mixed_list = []
#         images_flipped_list = []
#         pos_list = []
#         neg_list = []
        
#         image_mixed0, image_flipped0, pos, neg = mixup_filp_data(samples_origin[0], samples_new[0],alpha=0.5,direction=args.direction,ratio=args.ratio)
#         for image_o, image_n in zip(samples_origin, samples_new):
#             image_mixed, image_flipped, pos, neg = mixup_filp_data(image_o, image_n,alpha=0.5,direction=args.direction,ratio=args.ratio)
            
#             ### add some mask noise 
#             # noise_mask = torch.ones_like(image_mixed)
#             # # c,h,w = image_mixed.shape
#             # noise_mask[:, :, 20:38] = 0
#             # noise_mask[:, :, 90:108] = 0
#             # noise_mask[:, :, :64] = 0
#             # image_mixed_mask = image_mixed * noise_mask
#             # images_mixed_list.append(image_mixed_mask)

#             ### add mix 
#             # image_mixed[:, :, :32] = image_mixed0[:, :, :32]
#             # image_mixed[:, :, 96:] = image_mixed0[:, :, 96:]

#             ### add partial mix 
#             # image_mixed[:, :, :] = image_flipped[:, :, 80:]
#             # image_mixed[:, :, 64:] = image_flipped[:, :, 64:]

#             images_mixed_list.append(image_mixed)
#             images_flipped_list.append(image_flipped)
#             pos_list.append(pos)
#             neg_list.append(neg)

#         samples = torch.cat([t.unsqueeze(0) for t in images_mixed_list], 0)
#         samples = samples.to(device, non_blocking=True)
        
#         samples_flipped = torch.cat([t.unsqueeze(0) for t in images_flipped_list], 0)
#         samples_flipped = samples_flipped.to(device, non_blocking=True)
        
#         pos_prompt = torch.tensor(pos_list, dtype=torch.long).to(device, non_blocking=True)
#         neg_prompt = torch.tensor(neg_list, dtype=torch.long).to(device, non_blocking=True)
        
#         samples_origin = samples_origin.to(device, non_blocking=True)
#         samples_gt = {}
#         samples_gt.update({'pos':samples_origin})
#         samples_gt.update({'neg':samples_flipped})

#         loss, pred_pos, pred_neg = model(samples, samples_gt, pos_prompt, neg_prompt)

#         imgs_pred_pos = model.unpatchify(pred_pos) # [N,3,H,W]
#         imgs_pred_neg = model.unpatchify(pred_neg) # [N,3,H,W]

#         visualize_pair(imgs_pred_pos,imgs_pred_neg,samples,samples_origin,samples_flipped,args.demo_dir,(data_iter_step+1))

#         loss_value = loss.item()
#         loss_show += loss_value * samples_origin.shape[0]
#         if data_iter_step % accum_iter == 0:
#             print("iter step:{0},avg_loss:{1}".format(data_iter_step,loss/accum_iter))
        
#     print("total avg loss:", loss_show / nums)
    
def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    accum_iter = args.show_iter
    loss_show = 0
    nums = 0
    for data_iter_step, (samples_origin, _, _) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        nums += samples_origin.shape[0]
        if args.debug and data_iter_step == 50:
            print("stop testing at {}".format(data_iter_step))
            break

        images_mixed_list = []
        images_flipped_list = []
        pos_list = []
        neg_list = []
        
        for image_o in samples_origin:
            image_mixed, image_flipped, pos, neg = mixup_filp_data(image_o, alpha=0.5,direction=args.direction,ratio=args.ratio)
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

        pos_prompt = torch.tensor(pos_list, dtype=torch.long).to(device, non_blocking=True)
        neg_prompt = torch.tensor(neg_list, dtype=torch.long).to(device, non_blocking=True)
        
        samples_origin = samples_origin.to(device, non_blocking=True)
        samples_gt = {}
        samples_gt.update({'pos':samples_origin})
        samples_gt.update({'neg':samples_flipped})

        # loss, _, _ = model(samples, samples_gt, pos_prompt, neg_prompt)
        loss, pred_pos, pred_neg = model(samples, samples_gt, pos_prompt, neg_prompt)

        imgs_pred_pos = model.unpatchify(pred_pos) # [N,3,H,W]
        imgs_pred_neg = model.unpatchify(pred_neg) # [N,3,H,W]

        visualize_pair(imgs_pred_pos,imgs_pred_neg,samples,samples_origin,samples_flipped,args.demo_dir,(data_iter_step+1))

        loss_value = loss.item()
        loss_show += loss_value * samples_origin.shape[0]
        if data_iter_step % accum_iter == 0:
            print("iter step:{0},avg_loss:{1}".format(data_iter_step,loss/samples_origin.shape[0]))
        
    print("total avg loss:", loss_show / nums)

def train_one_epoch_old(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
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

    for data_iter_step, (samples_ori, samples_fli, samples_mix, pos_prompt, neg_prompt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if args.debug and data_iter_step == 500:
            print("stop training at {}".format(data_iter_step))
            break
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_new = copy.deepcopy(samples_origin)
        images_mixed_list = []
        images_flipped_list = []
        pos_list = []
        neg_list = []
        
        for image_o, image_n in zip(samples_origin, samples_new):
            image_mixed, image_flipped, pos, neg = mixup_filp_data(image_o, image_n,alpha=0.5,direction=args.direction,ratio=args.ratio)
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
        
        samples_gt = {}
        samples_gt.update({'pos':samples_origin})
        samples_gt.update({'neg':samples_flipped})
        
        
        with torch.cuda.amp.autocast():
            # loss, _ = model(samples, samples_origin, mask_ratio=args.mask_ratio)
            loss, _, _ = model(samples, samples_gt, pos_prompt, neg_prompt)

        loss_value = loss.item()
        # res_loss_value = res_loss.item()
        # align_loss_value = align_loss.item()

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
        # metric_logger.update(res_loss=res_loss_value)
        # metric_logger.update(align_loss=align_loss_value)


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

    for data_iter_step, (samples_origin, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if args.debug and data_iter_step == 50:
            print("stop training at {}".format(data_iter_step))
            break
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
    
        if args.direction!='rand':
            images_mixed_list = []
            images_flipped_list = []
            pos_list = []
            neg_list = []
            
            for image_o in samples_origin:
                image_mixed, image_flipped, pos, neg = mixup_filp_data(image_o, alpha=0.5,direction=args.direction,ratio=args.ratio)
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

            samples_gt = {}
            samples_gt.update({'pos':samples_origin})
            samples_gt.update({'neg':samples_flipped})

        else:
            bs = samples_origin.shape[0]
            indices = torch.randperm(bs)
            while (indices == torch.arange(bs)).any():
                indices = torch.randperm(bs)

            # 叠加图片
            # 使用索引选择要叠加的图片，并将它们叠加到原始图片上
            samples_add = samples_origin[indices]
            samples = 0.5* samples_origin + 0.5 * samples_add
            samples = samples.to(device, non_blocking=True)
            samples_add = samples_add.to(device, non_blocking=True)
            samples_origin = samples_origin.to(device, non_blocking=True)

            samples_gt = {}
            samples_gt.update({'pos':samples_origin})
            samples_gt.update({'neg':samples_add})

            pos_list = [0]*bs
            neg_list = [1]*bs

            pos_prompt = torch.tensor(pos_list, dtype=torch.long).to(device, non_blocking=True)
            neg_prompt = torch.tensor(neg_list, dtype=torch.long).to(device, non_blocking=True)

            # # 确保结果仍然在有效的像素值范围内
            # result = torch.clamp(result, 0, 1)

        if args.debug and data_iter_step % 20 == 0 :
            visualize_pair2(samples_origin,samples_add,samples,args.demo_dir,(data_iter_step+1))
        
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, samples_gt, pos_prompt, neg_prompt)

        loss_value = loss.item()
        # res_loss_value = res_loss.item()haha ai
        # align_loss_value = align_loss.item()

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
        # metric_logger.update(res_loss=res_loss_value)
        # metric_logger.update(align_loss=align_loss_value)


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

        # tensorbard log writer
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

         # wandb log writer
        if wandb_logger:
            wandb_logger._wandb.log(
                {'Rank-0 Batch Wise/train_loss': loss_value}, commit=False
            )
            wandb_logger._wandb.log({'Rank-0 Batch Wise/train_lr': lr}
            )
            

            # if class_acc:
            #     wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            # if use_amp:
            #     wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            # wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_test(model: torch.nn.Module,
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

    for data_iter_step, (student_input, teacher_input, pos_prompt, neg_prompt, if_reversed) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if args.debug and data_iter_step == 50:
            print("stop training at {}".format(data_iter_step))
            break

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            if args.mmschedule == 'const':
                mm = args.mm
            elif args.mmschedule == 'cosine':
                mm = 1. - 0.5 * (1. + math.cos(math.pi * (data_iter_step / len(data_loader) + epoch) / args.epochs)) * (1. - args.mm)
            metric_logger.update(mm=mm)
        update_mm = (data_iter_step % accum_iter == 0)

        student_mix, student_ori, student_fli = student_input
        student_mix = student_mix.to(device, non_blocking=True)
        student_fli = student_fli.to(device, non_blocking=True)
        student_ori = student_ori.to(device, non_blocking=True)
        # student_input = (student_mix, student_ori, student_fli)

        
        teacher_mix, teacher_ori, teacher_fli = teacher_input
        teacher_mix = teacher_mix.to(device, non_blocking=True)
        teacher_fli = teacher_fli.to(device, non_blocking=True)
        teacher_ori = teacher_ori.to(device, non_blocking=True)
        teacher_input = (teacher_mix, teacher_ori, teacher_fli)

        pos_prompt = pos_prompt.to(device, non_blocking=True)
        neg_prompt = neg_prompt.to(device, non_blocking=True)

        samples = student_mix
        samples_gt = {}
        samples_gt.update({'pos':student_ori})
        samples_gt.update({'neg':student_fli})
     
    
        # if args.debug and data_iter_step % 20 == 0 :
        # visualize_pair2(samples_origin,samples_add,samples,args.demo_dir,(data_iter_step+1))
        
        # dynamic ratio
        loss_feat_ratio = 0.0 if epoch < 10 else (epoch-9)/ (20*2)
        
        with torch.cuda.amp.autocast():
            loss, loss_outputs, _, _ = model(samples, samples_gt, teacher_mix, pos_prompt, neg_prompt,mm,update_mm,loss_feat_ratio)
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
        # metric_logger.update(res_loss=res_loss_value)
        # metric_logger.update(align_loss=align_loss_value)


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



def train_one_epoch_(model: torch.nn.Module,
                    teacher_model: torch.nn.Module,
                    teacher_model_without_ddp: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, 
                    patch_size: int = 16, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, momentum_schedule=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('res_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('align_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 60

 
    for step, (student_input, teacher_input, pos_prompt, neg_prompt, if_reversed) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if args.debug and step == 50:
            print("stop training at {}".format(step))
            break
    
        student_mix, student_ori, student_fli = student_input
        student_mix = student_mix.to(device, non_blocking=True)
        student_fli = student_fli.to(device, non_blocking=True)
        student_ori = student_ori.to(device, non_blocking=True)
        student_input = (student_mix, student_ori, student_fli)

        if args.use_online_target:
            teacher_mix, teacher_ori, teacher_fli = teacher_input
            teacher_mix = teacher_mix.to(device, non_blocking=True)
            teacher_fli = teacher_fli.to(device, non_blocking=True)
            teacher_ori = teacher_ori.to(device, non_blocking=True)
            teacher_input = (teacher_mix, teacher_ori, teacher_fli)
        else:
            teacher_input = None

        pos_prompt = pos_prompt.to(device, non_blocking=True)
        neg_prompt = neg_prompt.to(device, non_blocking=True)
     
        with torch.cuda.amp.autocast():
            loss = 0.
            #TODO：1. 通过use_ema判断是否使用moco损失 2. 如果使用可能需要内部处理两次
            out_dict = model(student_input, teacher_input, pos_prompt, neg_prompt)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, samples_gt, pos_prompt, neg_prompt)

        loss_value = loss.item()
        # res_loss_value = res_loss.item()haha ai
        # align_loss_value = align_loss.item()

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
        # metric_logger.update(res_loss=res_loss_value)
        # metric_logger.update(align_loss=align_loss_value)


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

        