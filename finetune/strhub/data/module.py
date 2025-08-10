# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import PurePath
from typing import Optional, Callable, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as T

from .dataset import build_tree_dataset, LmdbDataset, build_chinese_tree_dataset, ChineseLmdbDataset

class LengthBasedSampler(Sampler):
    def __init__(self, data_source, desired_length):
        self.data_source = data_source
        self.desired_length = desired_length

    def __iter__(self):
        # 获取所有满足长度条件的索引
        valid_indices = [i for i, (_, label) in enumerate(self.data_source) if len(label) > self.desired_length]
        return iter(valid_indices)

    def __len__(self):
        return len(self.valid_indices)

class SceneTextDataModule(pl.LightningDataModule):
    TEST_BENCHMARK_ART1_5 = ('wordartv1-test', 'wordartv1.5_train_lmdb')
    TEST_BENCHMARK_SUB = ('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80')
    TEST_BENCHMARK_ARD = ('IIIT5k', 'SVT', 'IC13_1015', 'IC15_1811', 'SVTP', 'CUTE80')
    TEST_BENCHMARK = ('IIIT5k', 'SVT', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80')
    TEST_NEW = ('TT', 'CTW', 'COCOv1.4','COCO2','ArT','Uber')
    # TEST_NEW = ('WORDART', 'data_shuffle', 'data_rand', 'data_shufflev2', 'data_randv2', 'HOST', 'WOST')
    TEST_UNION = ('artistic',  'contextless',  'curve',  'general',  'multi_oriented',  'multi_words',  'salient')
    TEST_ONLYCUS = ('TT', 'CTW', 'COCOv1.4','COCO2','ArT','Uber')
    # TEST_ONLYCUS = ('CTW', )
    TEST_FISYN = ('unseen_union_from_mjsttrain/lmdb', 'unseen_common_from_mjsttrain/lmdb')
    TEST_FIUNION = ('unseen_union_from_uniontrain/lmdb','unseen_common_from_uniontrain/lmdb')
    TEST_FIORI = ('unseen_union_from_origin/lmdb','unseen_common_from_origin/lmdb')
    TEST_UNILEN = ('union_benchmark_split/len_1', 'union_benchmark_split/len_2', 'union_benchmark_split/len_3', 'union_benchmark_split/len_4', 'union_benchmark_split/len_5', 'union_benchmark_split/len_6', 'union_benchmark_split/len_7', 'union_benchmark_split/len_8', 'union_benchmark_split/len_9', 'union_benchmark_split/len_10', 'union_benchmark_split/len_11', 'union_benchmark_split/len_12', 'union_benchmark_split/len_13', 'union_benchmark_split/len_14', 'union_benchmark_split/len_15', 'union_benchmark_split/len_16', 'union_benchmark_split/len_17', 'union_benchmark_split/len_18', 'union_benchmark_split/len_19', 'union_benchmark_split/len_20', 'union_benchmark_split/len_21', 'union_benchmark_split/len_22', 'union_benchmark_split/len_23', 'union_benchmark_split/len_24', 'union_benchmark_split/len_25', 'union_benchmark_split/len_26', 'union_benchmark_split/len_27', 'union_benchmark_split/len_28', 'union_benchmark_split/len_29', 'union_benchmark_split/len_30', 'union_benchmark_split/len_31', 'union_benchmark_split/len_32', 'union_benchmark_split/len_33', 'union_benchmark_split/len_34', 'union_benchmark_split/len_35', 'union_benchmark_split/len_36', 'union_benchmark_split/len_37', 'union_benchmark_split/len_38', 'union_benchmark_split/len_39', 'union_benchmark_split/len_40', 'union_benchmark_split/len_41', 'union_benchmark_split/len_42', 'union_benchmark_split/len_43', 'union_benchmark_split/len_44', 'union_benchmark_split/len_45', 'union_benchmark_split/len_46', 'union_benchmark_split/len_47', 'union_benchmark_split/len_48', 'union_benchmark_split/len_49', 'union_benchmark_split/len_50', 'union_benchmark_split/len_51', 'union_benchmark_split/len_52', 'union_benchmark_split/len_53', 'union_benchmark_split/len_54', 'union_benchmark_split/len_55', 'union_benchmark_split/len_56', 'union_benchmark_split/len_57', 'union_benchmark_split/len_58', 'union_benchmark_split/len_59', 'union_benchmark_split/len_60', 'union_benchmark_split/len_61', 'union_benchmark_split/len_62', 'union_benchmark_split/len_63', 'union_benchmark_split/len_65', 'union_benchmark_split/len_66', 'union_benchmark_split/len_67', 'union_benchmark_split/len_68', 'union_benchmark_split/len_69', 'union_benchmark_split/len_70', 'union_benchmark_split/len_71', 'union_benchmark_split/len_75', 'union_benchmark_split/len_76', 'union_benchmark_split/len_77', 'union_benchmark_split/len_80', 'union_benchmark_split/len_81', 'union_benchmark_split/len_83', 'union_benchmark_split/len_87', 'union_benchmark_split/len_89', 'union_benchmark_split/len_94', 'union_benchmark_split/len_101', 'union_benchmark_split/len_119', 'union_benchmark_split/len_161')
    TEST_COMLEN = ('common_benchmark_split/len_1','common_benchmark_split/len_2','common_benchmark_split/len_3','common_benchmark_split/len_4','common_benchmark_split/len_5', \
                   'common_benchmark_split/len_6','common_benchmark_split/len_7','common_benchmark_split/len_8','common_benchmark_split/len_9','common_benchmark_split/len_10', \
                    'common_benchmark_split/len_11','common_benchmark_split/len_12','common_benchmark_split/len_13','common_benchmark_split/len_14','common_benchmark_split/len_15', \
                    'common_benchmark_split/len_16', 'common_benchmark_split/len_17','common_benchmark_split/len_18','common_benchmark_split/len_19','common_benchmark_split/len_20', \
                    'common_benchmark_split/len_21', 'common_benchmark_split/len_22','common_benchmark_split/len_23','common_benchmark_split/len_24','common_benchmark_split/len_25' )
    TEST_ALL = tuple(set(TEST_BENCHMARK_SUB + TEST_BENCHMARK + TEST_NEW + TEST_UNION))

    def __init__(self, root_dir: str, train_dir: str, img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, augment: bool,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None
    
    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
        transforms = []
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            if self.train_dir == 'Union':
                root = PurePath(self.root_dir, self.train_dir,'train_union')
            elif self.train_dir == 'icdar24':
                root = PurePath(self.root_dir, self.train_dir,'train')
            elif self.train_dir == 'iam':
                root = PurePath(self.root_dir, self.train_dir,'iam_train')
            elif self.train_dir == 'cvl':
                root = PurePath(self.root_dir, self.train_dir,'cvl_train')
            else:
                root = PurePath(self.root_dir, 'train', self.train_dir)

            self._train_dataset = build_tree_dataset(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            if self.train_dir == 'Union':
                root = PurePath(self.root_dir, self.train_dir,'val_union')
            if self.train_dir == 'icdar24':
                root = PurePath(self.root_dir, self.train_dir,'val')
            elif self.train_dir == 'iam':
                root = PurePath(self.root_dir, self.train_dir,'iam_test')
            elif self.train_dir == 'cvl':
                root = PurePath(self.root_dir, self.train_dir,'cvl_test')
            else:
                root = PurePath(self.root_dir, 'val')
            self._val_dataset = build_tree_dataset(root, self.charset_test, self.max_label_length,
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   transform=transform)
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
    
    def test_custom_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test_custom')
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
    

    #sampler = LengthBasedSampler(dataset, desired_length)

    def test_ood_dataloaders(self, subset, desired_length):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        # sampler = LengthBasedSampler(dataset, desired_length)
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn, sampler=LengthBasedSampler(v, desired_length))
                for k, v in datasets.items()}
    

class ChineseSceneTextDataModule(pl.LightningDataModule):
    TEST_WEB = ('web_test',)
    TEST_SCENE = ('scene_test',)
    TEST_DOC = ('document_test',)

    # TEST_ALL = tuple(set(TEST_BENCHMARK_SUB + TEST_BENCHMARK + TEST_NEW + TEST_UNION))

    def __init__(self, root_dir: str, train_dir: str, img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, augment: bool,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None

        print("***********traindir:",self.train_dir)
    
    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
        transforms = []
        if augment:
            from .augment import rand_augment_transform
            transforms.append(rand_augment_transform())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            if self.train_dir == 'scene_train':
                root = PurePath(self.root_dir, 'scene', self.train_dir)
            elif self.train_dir == 'web_train':
                root = PurePath(self.root_dir, 'web', self.train_dir)
            elif self.train_dir == 'document_train':
                root = PurePath(self.root_dir, 'document', self.train_dir)
            
            self._train_dataset = build_chinese_tree_dataset(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            if self.train_dir == 'scene_train':
                root = PurePath(self.root_dir, 'scene', 'scene_val')
            elif self.train_dir == 'web_train':
                root = PurePath(self.root_dir, 'web', 'web_val')
            elif self.train_dir == 'document_train':
                root = PurePath(self.root_dir, 'document', 'document_val')
            
            self._val_dataset = build_chinese_tree_dataset(root, self.charset_test, self.max_label_length,
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   transform=transform)
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: ChineseLmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}



if __name__=='__main__':
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler

    class LengthBasedSampler(Sampler):
        def __init__(self, data_source, desired_length):
            self.data_source = data_source
            self.desired_length = desired_length

        def __iter__(self):
            # 获取所有满足长度条件的索引
            valid_indices = [i for i, (_, label) in enumerate(self.data_source) if len(label) == self.desired_length]
            return iter(valid_indices)

        def __len__(self):
            return len(self.valid_indices)

    class MyDataset(Dataset):
        def __init__(self):
            self.data = [
                ("data1", "label1"),
                ("data2", "lab2"),
                ("data3", "labl3"),
                ("data4", "lbl4"),
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # 使用自定义sampler来选择标签长度为5的数据
    desired_length = 5
    dataset = MyDataset()
    sampler = LengthBasedSampler(dataset, desired_length)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    for batch in dataloader:
        print(batch)
