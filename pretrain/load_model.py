import torch
# /home/gaoz/output/dig-cc/
path = '/home/gaoz/output/dig-cc/checkpoint-9.pth'
checkpoint = torch.load(path, map_location='cpu')
pretrained_dict = checkpoint['model']
# print(pretrained_dict.keys())

print('-' * 100)

path = '/home/gaoz/output_pixel/SR-100e-noaug_norm/checkpoint-167.pth'
model_checkpoint = torch.load(path, map_location='cpu')
model_dict = model_checkpoint['model']
# print(model_dict.keys())


pretrained_dict_new = {k[8:]: v for k, v in pretrained_dict.items() \
                                if k.startswith('encoder') and (k[8:] in model_dict)}#filter out unnecessary keys  and (k[8:] != '.pos_embed')

for k, v in pretrained_dict.items():
    if k.startswith('encoder'):
        print(k[8:])
    


print(len(pretrained_dict_new.keys()))