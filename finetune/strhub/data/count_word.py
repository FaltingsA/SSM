import lmdb
import os
import json

## 过滤一遍train，记录到json中

# settings
path_list = ['/home/sist/zuangao/datasets/unidata/data/train/real/OpenVINO/train-2', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/OpenVINO/train_1', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/OpenVINO/train_5', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/OpenVINO/train_f', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/TextOCR/train', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/TextOCR/val', \
             '/home/sist/zuangao/datasets/unidata/data/train/real/MLT19/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/MLT19/test',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/MLT19/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/RCTW17/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/RCTW17/test',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/RCTW17/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/COCOv2.0/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/COCOv2.0/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/ArT/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/ArT/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/Uber/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/Uber/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/LSVT/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/LSVT/test',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/LSVT/val',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/ReCTS/train',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/ReCTS/test',\
             '/home/sist/zuangao/datasets/unidata/data/train/real/ReCTS/val']

data_root = ''
save_root = '/home/sist/zuangao/datasets/unidata/data/test_custom/unseen_origin/'
os.makedirs(save_root,exist_ok=True)
map_size=30073741824
max_length = 25

## json: 统计单词出现的频率
record_json = os.path.join(save_root, 'wordfre_json') 
f = open(record_json,'w',encoding='utf-8')

## 先进行长度统计

word_fre = {}

## 统计所有数据样本
cnt = 0
for data_path in path_list:
    data_path = os.path.join(data_root,data_path)
    env = lmdb.open(data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        cache = {}

        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            length = len(label)
            
            if label in word_fre:
                word_fre[label]+=1
            else:
                word_fre[label] = 1

            if cnt % 5000==0:
                print(cnt,'cur:',label, length)
            cnt += 1

json.dump(word_fre, f)
            

