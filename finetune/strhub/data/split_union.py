import lmdb
import os
import json

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

# settings
path_list = ['salient','artistic','contextless','multi_words','multi_oriented','curve','general']
# path_list = ['general']
# # path_list = ['CUTE80']
# data_root = '/home/gaoz/datasets/unidata/data/testp/union-benchmark'
# save_root = '/home/gaoz/datasets/unidata/data/union_benchmark_split/'

# path_list = ['IC13_857','IC13_1015','IC15_1811','IC15_2077','IIIT5k','SVT','SVTP','CUTE80']

# path_list = ['CUTE80']
data_root = '/home/sist/zuangao/datasets/unidata/data/test'
save_root = '/home/sist/zuangao/datasets/unidata/data/test_custom/union_benchmark_split/'

os.makedirs(save_root,exist_ok=True)
map_size=40073741824
max_length = 162

## json
record_json1 = os.path.join(save_root,'record1.json' )
record_json2 = os.path.join(save_root,'record2.json' )
f1 = open(record_json1,'w',encoding='utf-8')
f2 = open(record_json2,'w',encoding='utf-8')

## 先整体进行长度统计

length_cnt = {i: 0 for i in range(max_length+2)} # 0 1 2 ..... max_length max_length+1

len_31 = []
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
            if length == 31:len_31.append((index,label))
            if len(label) < 1:
                print('no label:',index)
                length_cnt[length] +=1

            if len(label) > max_length:
                length_cnt[max_length+1] +=1
                
            
            else:
                length_cnt[length] +=1

print(len(len_31), len_31)            
length_cnt = {k: v for k, v in length_cnt.items() if v != 0}
# for k,v in length_cnt.items():
#     print(k, v)

json.dump(length_cnt, f1)
## 根据长度创建多个写入
env_dict = {}
for k,_ in length_cnt.items():
    if k==0:continue
    target = 'len_'+str(k)
    save_path = os.path.join(save_root, target)
    env = lmdb.open(save_path, map_size=map_size)
    env_dict.update({k:env}) #1,2,.......26

## 统计所有数据样本
n= 0 
cnt = 0
length_cnt_new = {k:0 for k in length_cnt.keys()}
for data_path in path_list:
    data_path = os.path.join(data_root,data_path)
    env = lmdb.open(data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        n+=nSamples
        # 维护一个cache_list_dict
        cache_list_dict = {}

        for k,_ in length_cnt.items():
            if k==0:continue
            cache_list_dict.update({k:{}}) #1,2,.......26

        # 维护一个 sub_cnt_dict
        sub_cnt_dict = {}
        for k,_ in length_cnt.items():
            if k==0:continue
            sub_cnt_dict.update({k:0}) #1,2,.......26

        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            
            length = len(label)

            if len(label) < 1:
                print('no label:',index)
               

            if len(label) > max_length:
                length_cnt_new[max_length+1] +=1
                save_index = length_cnt_new[max_length+1]
            
            else:
                length_cnt_new[length] +=1
                save_index = length_cnt_new[length]

            # env = env_dict[length]

            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            
            if len(label) == 31: print(label,len(label))
            imageKey = 'image-%09d'.encode() % save_index
            labelKey = 'label-%09d'.encode() % save_index

            # cache[imageKey] = imgbuf
            # cache[labelKey] = label.strip().encode()

            cache_list_dict[length][imageKey] = imgbuf
            cache_list_dict[length][labelKey] = label.strip().encode()

            ## 当某个length的子统计到达1000
            if  sub_cnt_dict[length] % 1000 == 0:
                newenv = env_dict[length]
                writeCache(newenv, cache_list_dict[length])
                cache_list_dict[length] = {}
                # print('Written 1000')

            sub_cnt_dict[length] +=1
            cnt += 1

        # print(data_path)
        # print('*' * 50)
        # print(sub_cnt_dict[31])
        # for 
        # print(cache_list_dict[31][labelKey])
        # print('*' * 50)
        ## 每遍历一个子集, 对每个length如果没满1000 也存一下
        for k,v in sub_cnt_dict.items():
            newenv = env_dict[k]
            writeCache(newenv, cache_list_dict[k])
        
        # sub_cnt_dict = {}
        print(data_path, nSamples,'finsh')



vs = 0
for k,v in length_cnt_new.items():
    if k==0: continue
    # print(k, v)
    newenv = env_dict[k]
    cache['num-samples'.encode()] = str(v).encode()
    writeCache(newenv, cache)
    vs+=v
    
print("total number: ", cnt,n,vs)
if length_cnt_new == length_cnt:
    json.dump(length_cnt, f2)
    # print('error')
else:
    print('error')
    json.dump(length_cnt_new, f2)
            

# r = '/home/sist/zuangao/datasets/unidata/data/test_custom/union_benchmark_split/'
# ps = [ ('union_benchmark_split/' + i) for i in os.listdir(r) if i.startswith('len_') ]
# ps = tuple(ps)
# print(ps)


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
