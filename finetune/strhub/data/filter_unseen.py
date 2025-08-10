import lmdb
import os
import json

def save_dict_as_json(data_dict, filename):
    with open(filename, 'w') as file:
        json.dump(data_dict, file)

def load_dict_from_json(filename):
    with open(filename, 'r') as file:
        data_dict = json.load(file)
    return data_dict

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

# settings
path_list = ['salient','artistic','contextless','multi_words','multi_oriented','curve','general']

# data_root = '/home/gaoz/datasets/unidata/data/testp/union-benchmark'
# path_list = ['IC13_857','IC13_1015','IC15_1811','IC15_2077','IIIT5k','SVT','SVTP','CUTE80']

data_root = '/home/sist/zuangao/datasets/unidata/data/test'
save_root = '/home/sist/zuangao/datasets/unidata/data/unseen_union_from_origin/'
os.makedirs(save_root,exist_ok=True)
map_size=30073741824
max_length = 25

## json
union_type = False
unseen_json = os.path.join(save_root,'unseen.json')
union_wordfre_json = '/home/sist/zuangao/datasets/unidata/data/test_custom/unseen_union/wordfre_json'
# mjst_wordfre_json = '/home/sist/zuangao/datasets/unidata/data/test_custom/unseen_mjst/wordfre_json'
origin_wordfre_json = "/home/sist/zuangao/datasets/unidata/data/test_custom/unseen_origin/wordfre_json"
# dict 
unseen = {}

if union_type:
    wordfre_dict = load_dict_from_json(union_wordfre_json)

else:
    wordfre_dict = load_dict_from_json(origin_wordfre_json)

save_path = os.path.join(save_root, 'lmdb')
newenv  = lmdb.open(save_path, map_size=map_size)

## 统计所有数据样本
cnt = 1
for path in path_list:
    data_path = os.path.join(data_root,path)
    env = lmdb.open(data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        cache = {}
        
        sub_cnt = 0
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            length = len(label)
            
            # label is not unseen
            if label in wordfre_dict: continue
            
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imgbuf
            cache[labelKey] = label.strip().encode()
            
            unseen[label]=length
            
            if sub_cnt % 1000 == 0:
                writeCache(newenv, cache)
                cache = {}
                print('Written 1000')
           
            cnt += 1
            sub_cnt +=1


        ## 如果没满1000 也存一下
        writeCache(newenv, cache)
        print(data_path,nSamples,sub_cnt)

print("total number: ", cnt)

i = cnt - 1
cache['num-samples'.encode()] = str(i).encode()
writeCache(newenv, cache)
print('Created dataset with %d samples' % i)

save_dict_as_json(unseen,unseen_json)


# class LmdbDataset(Dataset):

#     def __init__(self, root, opt):

#         self.root = root
#         self.opt = opt
#         self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
#         if not self.env:
#             print('cannot create lmdb from %s' % (root))
#             sys.exit(0)

#         with self.env.begin(write=False) as txn:
#             nSamples = int(txn.get('num-samples'.encode()))
#             self.nSamples = nSamples
#             if self.opt.data_filtering_off:
#                 # for fast check or benchmark evaluation with no filtering
#                 self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
#             else:
#                 """ Filtering part
#                 If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
#                 use --data_filtering_off and only evaluate on alphabets and digits.
#                 see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

#                 And if you want to evaluate them with the model trained with --sensitive option,
#                 use --sensitive and --data_filtering_off,
#                 see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
#                 """
#                 self.filtered_index_list = []
#                 for index in range(self.nSamples):
#                     index += 1  # lmdb starts with 1
#                     label_key = 'label-%09d'.encode() % index
#                     label = txn.get(label_key).decode('utf-8')
#                     if len(label) > self.opt.max_length:
#                         continue
#                     self.filtered_index_list.append(index)

#                 self.nSamples = len(self.filtered_index_list)
                
#     def __len__(self):
#         return self.nSamples

#     def __getitem__(self, index):
#         assert index <= len(self), 'index range error'
#         index = self.filtered_index_list[index]

#         with self.env.begin(write=False) as txn:
#             label_key = 'label-%09d'.encode() % index
#             label = txn.get(label_key).decode('utf-8')
#             img_key = 'image-%09d'.encode() % index
#             imgbuf = txn.get(img_key)

#             buf = six.BytesIO()
#             buf.write(imgbuf)
#             buf.seek(0)
#             try:
#                 if self.opt.rgb:
#                     img = Image.open(buf).convert('RGB')  # for color image
#                 else:
#                     img = Image.open(buf).convert('L')

#             except IOError:
#                 print(f'Corrupted image for {index}')
#                 # make dummy image and dummy label for corrupted image.
#                 if self.opt.rgb:
#                     img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
#                 else:
#                     img = Image.new('L', (self.opt.imgW, self.opt.imgH))
#                 label = '[dummy_label]'

#             if not self.opt.sensitive:
#                 label = label.lower()

#             # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
#             out_of_char = f'[^{self.opt.character}]'
#             label = re.sub(out_of_char, '', label)

#         return (img, label)
