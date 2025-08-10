#!/usr/bin/env python3
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

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
import os
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from strhub.data.dataset import PathLmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--union', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--standard', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    kwargs.update({'standard': args.standard})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)
    '''
    # test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK_ARD + SceneTextDataModule.TEST_BENCHMARK
    test_set = SceneTextDataModule.TEST_BENCHMARK_ART1_5
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    if args.union:
        test_set += SceneTextDataModule.TEST_UNION
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_ART1_5
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    if args.union:
        result_groups.update({'Union': SceneTextDataModule.TEST_UNION})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)
    '''
#===============================================
    res_out = './testB-SSMv2-res.txt'
    res_conf_out = './testB-SSMv2-res-conf.txt'
    fr = open(res_out,'w')
    frc = open(res_conf_out,'w')
    transform = datamodule.get_transform(datamodule.img_size, rotation=datamodule.rotation)
    testA_root = '/home/gaoz/datasets/icdar24/test/wordartv1.5_testB_lmdb'

    testA_dataset = PathLmdbDataset(testA_root, datamodule.charset_test, datamodule.max_label_length,
                                    datamodule.min_image_dim, datamodule.remove_whitespace, datamodule.normalize_unicode,
                                    transform=transform) 
    testA_loader = DataLoader(testA_dataset, batch_size=datamodule.batch_size, num_workers=datamodule.num_workers,
                                pin_memory=True, collate_fn=datamodule.collate_fn)
    # print(os.listdir(testA_root), len(testA_loader))
    
    for images, label, path in tqdm(testA_loader,total=len(testA_loader)):
        # print(images.shape, label, path)
        path = path[0]
        images = images.to(args.device)
        logits = model.forward(images)
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)

        for pred, prob, in zip(preds, probs):
            confidence = prob.prod().item()
            pred = model.charset_adapter(pred)
            fr_info = f'{path} {pred}\n'
            frc_info = f'{path} {pred} {confidence}\n'
            # print(frc_info)
            fr.write(fr_info)
            frc.write(frc_info)

    fr.close()
    frc.close()

if __name__ == '__main__':
    main()
