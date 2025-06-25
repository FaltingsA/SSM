# SSM
[IJCAI-2024] The official code of  Self-Supervised Pre-training with Symmetric Superimposition Modeling for Scene Text Recognition

## Model weights
[Google Cloud](https://drive.google.com/drive/folders/1_tGsCcUNiWf_hgE43dzOy2yM5nYW7UUu?usp=drive_link)

## Datasets
Download the [datasets](Datasets.md) from the following links:
1. [LMDB archives](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.
2. [LMDB archives](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) for TextOCR and OpenVINO.
3. [Union14M archives](https://github.com/Mountchicken/Union14M/blob/main/docs/source_dataset.md)for Union14M

## Pretraining
1. Place your data in data_path and create output_path directory
2. Execute the following script:
```sh
cd pretrain
bash scripts/encoder-pretrain.sh
```
The shell script contains the following configuration:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --master_port 29060 sim_pretrain.py  \
--lr 5e-4 \
--batch_size 144 \
--mode single \
--model flipae_sim_vit_small_str \
--epochs 20 \
--warmup_epochs 1  \
--mm 0.995 \
--mmschedule 'cosine' \
--output_dir $output_path$ \
--data_path $data_path$ \
--direction aug_pool --num_workers 10 
```

## Finetuning
```sh
cd finetune
CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.py +experiment=mdr-dec6-union
```
### Finetune using pretrained weights
```bash
./train.py +experiment=parseq-tiny pretrained=parseq-tiny  # Not all experiments have pretrained weights
```

### Train a model variant/preconfigured experiment
The base model configurations are in `configs/model/`, while variations are stored in `configs/experiment/`.
```bash
./train.py +experiment=parseq-tiny  # Some examples: abinet-sv, trbc
```

### Specify the character set for training
```bash
./train.py charset=94_full  # Other options: 36_lowercase or 62_mixed-case. See configs/charset/
```

### Specify the training dataset
```bash
./train.py dataset=real  # Other option: synth. See configs/dataset/
```

### Change general model training parameters
```bash
./train.py model.img_size=[32, 128] model.max_label_length=25 model.batch_size=384
```

### Change data-related training parameters
```bash
./train.py data.root_dir=data data.num_workers=2 data.augment=true
```

### Change `pytorch_lightning.Trainer` parameters
```bash
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2
```
Note that you can pass any [Trainer parameter](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html),
you just need to prefix it with `+` if it is not originally specified in `configs/main.yaml`.

### Resume training from checkpoint (experimental)
```bash
./train.py +experiment=<model_exp> ckpt_path=outputs/<model>/<timestamp>/checkpoints/<checkpoint>.ckpt
```

</p></details>

## Evaluation
The test script, ```test.py```, can be used to evaluate any model trained with this project. For more info, see ```./test.py --help```.

PARSeq runtime parameters can be passed using the format `param:type=value`. For example, PARSeq NAR decoding can be invoked via `./test.py parseq.ckpt refine_iters:int=2 decode_ar:bool=false`.

### Lowercase alphanumeric comparison on benchmark datasets 
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt  
```

<details><summary>Sample commands for reproducing results</summary><p>

### Benchmark using different evaluation character sets
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt  # lowercase alphanumeric (36-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased  # mixed-case alphanumeric (62-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation  # mixed-case alphanumeric + punctuation (94-character set)
```

### Lowercase alphanumeric comparison on more challenging datasets 
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --new
```

## ✍️ Citation

```bibtex
@misc{gao2024selfsupervised,
      title={Self-Supervised Pre-training with Symmetric Superimposition Modeling for Scene Text Recognition}, 
      author={Zuan Gao and Yuxin Wang and Yadong Qu and Boqiang Zhang and Zixiao Wang and Jianjun Xu and Hongtao Xie},
      year={2024},
      eprint={2405.05841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

