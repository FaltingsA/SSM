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
from util.transforms import CVColorJitter, CVDeterioration, CVGeometry
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils

from imgaug import augmenters as iaa

import random

def set_seed(seed):
    """设置全局随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)  # 示例种子

def get_fixed_transform_affine(seed=42):
   # 定义固定的随机种子
    random_seed = torch.manual_seed(seed)
    # 定义旋转和透视变换
    affine_transform = transforms.RandomAffine(translate=(0,0.1), degrees=(-10, 10), shear=(-25,-25,-10, 10), fill=0)
    graystyle_transform = transforms.RandomGrayscale(p=0.2)

    # 返回Compose变换
    return transforms.Compose([
        affine_transform,
        graystyle_transform
    ])
    
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

def hierarchical_pixel_dataset(root, args):
    # try:
    #     kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    # except KeyError:
    #     pass
    root = Path(root).absolute()
    dataset_log = f'dataset root:\t{root}\n'
    print(f'dataset root:\t{root}')
    datasets = []
    if args.pixel_type=='SR':
        datapipe = SRLmdbDataset
        for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
            mdb = Path(mdb)
            ds_name = str(mdb.parent.relative_to(root))
            ds_root = str(mdb.parent.absolute())
            print(ds_name,ds_root)
            dataset = datapipe(ds_root, args)
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
            self.camera = [Contrast(), Brightness()]

            self.pattern = [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]

            self.noise = [GaussianNoise()]
            self.blur = [GaussianBlur(), MotionBlur()]
            # self.weather = [Fog(), Frost(), Rain(), Shadow()]

            self.noises = [self.blur, self.noise,]
            self.processes = [self.camera, self.process]

            self.warp = [Curve(), Distort(), Stretch()]
            self.geometry = [Rotate(), Perspective(), Shrink()]

            self.isbaseline_aug = False
            # rand augment
            if self.opt.isrand_aug:
                # remove geometry warp
                self.augs = [self.process, self.camera, self.noise, self.blur]
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

        # img = transforms.ToTensor()(img)
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

class IrrDataAugment(object):
    '''
    Supports with and without data augmentation 
    '''
    def __init__(self, opt):
        self.opt = opt

        # self.mix = [Solarize(), Invert(), GaussianBlur(), Sharpness()]
        # self.geometry = [Rotate(), Perspective(), Shrink()]
        # self.warp = [Curve(), Distort()]
        # self.augs = [self.mix, self.warp, self.geometry]
        self.geometry = [Rotate(),Perspective(), Shrink()]
        self.augs = [self.geometry]

        self.scale = False

    def __call__(self, img1, img2):
        '''
            Must call img.copy() if pattern, Rain or Shadow is used
        '''
        img1, img2 = self.rand_aug(img1, img2)
        # img = transforms.ToTensor()(img)
        return img1, img2

    def rand_aug(self, img1, img2):
        aug = self.augs[0]
        
        index = np.random.randint(0, len(aug))
        op = aug[index]
        mag = np.random.randint(0, 3) if self.opt.augs_mag is None else self.opt.augs_mag
        
        img1 = op(img1, mag=mag)
        img2 = op(img2, mag=mag)

        return img1, img2

def get_rand_seed():
    seed = random.randint(0,10000)
    return seed

class DataAugment_weak(object):
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
                self.augs = [self.process, self.camera, self.noise, self.blur, self.weather, self.pattern, self.warp]
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
        
class AlignCollate_test(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt
        self.Hflip = transforms.RandomHorizontalFlip(p=opt.ratio)
        self.Vflip = transforms.RandomVerticalFlip(p=opt.ratio)
        self.Srotate = transforms.RandomRotation((180,180),expand=True)
        
    def get_aug_op(self):
        direction=self.opt.direction
        if_reversed = True
        if direction=='HF':
            pos = 0
            neg = 1
            aug_op = self.Hflip

        elif direction=='VF':
            pos = 0
            neg = 2
            aug_op = self.Vflip
            if_reversed = False

        elif direction=='Hybrid':
            prob = random.random()
            if prob > 0.5:
                pos = 0
                neg = 1
                aug_op = self.Hflip
            else:
                pos = 0
                neg = 2
                aug_op = self.Vflip
                if_reversed = False

        elif direction == 'aug_pool':
            prob = random.random()
            if prob > 0.66:
                pos = 0
                neg = 1
                aug_op = self.Hflip
            elif prob > 0.33:
                pos = 0
                neg = 2
                aug_op = self.Vflip
                if_reversed = False
            else:
                pos = 0
                neg = 3
                aug_op = self.Srotate

        return aug_op, pos, neg, if_reversed
    
    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        if self.opt.eval_img:
            images, labels, img_paths = zip(*batch)
        else:
            images, labels = zip(*batch)
            img_paths = None

        pil = transforms.ToPILImage()
        aug_transform = DataAugment_weak(self.opt)
        # image_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images]
        # ----------------------
        student_ori = []
        student_fli = []
        student_mix = []
        pos_prompt = []
        neg_prompt = []
        teacher_ori = []
        teacher_fli = []
        teacher_mix = []
        # ----------------------
        # get aug_op & prompt for one batch
        # aug_op, pos, neg, if_reversed = self.get_aug_op()
        for idx in range(len(images)):
            aug_op, pos, neg, if_reversed = self.get_aug_op()
            pos_prompt.append(pos)
            neg_prompt.append(neg)

            image = images[idx].resize((self.imgW, self.imgH), Image.BICUBIC)
            st_ori = aug_transform(image)
            st_fli = aug_op(st_ori)
            st_mix = 0.5 * (st_ori + st_fli)
                
            ta_ori = aug_transform(image)
            ta_fli = aug_op(ta_ori)
            # rotation，shear，and so on
            #-------------------------------------------------
            fix_affine_aug = get_fixed_transform_affine(idx)
            ta_ori = fix_affine_aug(ta_ori)
            fix_affine_aug = get_fixed_transform_affine(idx)
            ta_fli = fix_affine_aug(ta_fli)
            #-------------------------------------------------
            ta_mix = 0.5 * (ta_ori + ta_fli)
               
            student_ori.append(st_ori)
            student_fli.append(st_fli)
            student_mix.append(st_mix)

            teacher_ori.append(ta_ori)
            teacher_fli.append(ta_fli)
            teacher_mix.append(ta_mix) 
        

        student_ori = torch.cat([t.unsqueeze(0) for t in student_ori], 0)
        student_fli = torch.cat([t.unsqueeze(0) for t in student_fli], 0)
        student_mix = torch.cat([t.unsqueeze(0) for t in student_mix], 0)
        student_batch = (student_mix, student_ori, student_fli)
        
        teacher_ori = torch.cat([t.unsqueeze(0) for t in teacher_ori], 0)
        teacher_fli = torch.cat([t.unsqueeze(0) for t in teacher_fli], 0)
        teacher_mix = torch.cat([t.unsqueeze(0) for t in teacher_mix], 0)
        teacher_batch = (teacher_mix, teacher_ori, teacher_fli)
        
        pos_prompt = torch.tensor(pos_prompt, dtype=torch.long)
        neg_prompt = torch.tensor(neg_prompt, dtype=torch.long)
       

        return student_batch, teacher_batch, pos_prompt, neg_prompt, if_reversed


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
        aug_transform = DataAugment_weak(self.opt)
        
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

            ta_image_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images]
            ta_image_tensors = torch.cat([t.unsqueeze(0) for t in ta_image_tensors], 0)

        return image_tensors, ta_image_tensors


#==============================================

class SRLmdbDataset(Dataset):
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

                
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index += 1
        assert index <= len(self), 'index range error'

        with self.env.begin(write=False) as txn:
            img_lr_key = 'image_lr-%09d'.encode() % index
            imgbuf_lr = txn.get(img_lr_key)

            buf1 = six.BytesIO()
            buf1.write(imgbuf_lr)
            buf1.seek(0)
             
            img_lr = Image.open(buf1).convert('RGB')  # for color image
            
            img_hr_key = 'image_hr-%09d'.encode() % index
            imgbuf_hr = txn.get(img_hr_key)

            buf2 = six.BytesIO()
            buf2.write(imgbuf_hr)
            buf2.seek(0)
            img_hr = Image.open(buf2).convert('RGB')

        return (img_lr, img_hr)

class SRAlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        # if self.opt.eval_img:
        images_lr, images_hr = zip(*batch)
       
        if self.opt.eval_only or self.opt.eval:
            # aug_transform = transforms.Compose([
            #     transforms.ToTensor()
            # ])
            aug_transform = DataAugment_weak(self.opt)
        else:
            aug_transform = DataAugment_weak(self.opt)
            
        
        set_seed(42)
        image_lr_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images_lr]
        image_lr_tensors = torch.cat([t.unsqueeze(0) for t in image_lr_tensors], 0)
        set_seed(42)
        image_hr_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images_hr]
        image_hr_tensors = torch.cat([t.unsqueeze(0) for t in image_hr_tensors], 0)

        return image_lr_tensors, image_hr_tensors


class TextSegmentationDataset(Dataset):
    def __init__(self, root, args):
        """
        Args:
            image_dir (string): 图像文件夹的路径。
            mask_dir (string): Mask 图像文件夹的路径。
            transform (callable, optional): 应用于图像和mask的可选转换。
        """
        image_dir = os.path.join(root,'image')
        mask_dir = os.path.join(root,'mask')
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.jpg"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # 确保mask是单通道的

        # if self.transform:
        #     image = self.transform(image.resize((self.imgW, self.imgH), Image.BICUBIC))
        #     mask = self.transform(mask.resize((self.imgW, self.imgH), Image.BICUBIC))

        return (image, mask)


class SegAlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        # if self.opt.eval_img:
        images, maskes = zip(*batch)
       

        # pil = transforms.ToPILImage()
        if self.opt.eval_only or self.opt.eval:
            # aug_transform = transforms.Compose([
            #     transforms.ToTensor()
            # ])
            aug_transform = DataAugment_weak(self.opt)
        else:
            aug_transform = DataAugment_weak(self.opt)
        
        set_seed(42)
        image_tensors = [aug_transform(image.resize((self.imgW, self.imgH), Image.BICUBIC)) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        set_seed(42)
        mask_tensors = [aug_transform(mask.resize((self.imgW, self.imgH), Image.BICUBIC)) for mask in maskes]
        mask_tensors = torch.cat([t.unsqueeze(0) for t in mask_tensors], 0)

        return image_tensors, mask_tensors

# if __name__ == '__main__':
