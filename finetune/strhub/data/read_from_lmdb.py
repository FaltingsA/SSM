import lmdb
import os
import json
from PIL import Image
import re
import six
import numpy as np

import re

def get_trailing_number(s):
    # 使用正则表达式查找字符串末尾的数字
    match = re.search(r'(\d+)$', s)
    return int(match.group()) if match else 0

strings = ["item12", "item2", "item112", "item11"]
sorted_strings = sorted(strings, key=get_trailing_number)
print(sorted_strings)
# 输出: ['item2', 'item11', 'item12', 'item112']


r = '/home/sist/zuangao/datasets/unidata/data/test_custom/union_benchmark_split/'
ps = [ ('union_benchmark_split/' + i) for i in os.listdir(r) if i.startswith('len_') ]
ps = sorted(ps, key=get_trailing_number)
ps = tuple(ps)
print(ps)

# # settings
# path_list = ['len_31']
# data_root = '/home/sist/zuangao/datasets/unidata/data/test_custom/union_benchmark_split/'
# save_root = '/home/sist/zuangao/datasets/unidata/data/union_benchmark_split/test_split/'

# os.makedirs(save_root,exist_ok=True)
# map_size=30073741824
# max_length = 25

# data_path = os.path.join(data_root, path_list[0])
# # data_path = '/home/sist/zuangao/datasets/unidata/data/test_custom/filter_union_from_uniontrain/lmdb'
# env = lmdb.open(data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
# print(data_path)
# print(env)

# ct = 0
# with env.begin(write=False) as txn:
#     nSamples = int(txn.get('num-samples'.encode()))
#     print('get in')
#     print(nSamples)
#     cache = {}

#     for index in range(nSamples):
#         index += 1  # lmdb starts with 1
#         label_key = 'label-%09d'.encode() % index
#         label = txn.get(label_key).decode('utf-8')

#         print(ct,label,len(label))
#         img_key = 'image-%09d'.encode() % index
#         imgbuf = txn.get(img_key)

#         buf = six.BytesIO()
#         buf.write(imgbuf)
#         buf.seek(0)
#         try:
#             img = Image.open(buf).convert('RGB')  # for color image
           

#         except IOError:
#             print(f'Corrupted image for {index}')
#             # make dummy image and dummy label for corrupted image.
#             img = Image.new('RGB', (32, 128))
           
#             label = '[dummy_label]'

       
#         img.save(os.path.join(save_root, str(index)+'.jpg'))
#         ct += 1
#         if ct==100:break