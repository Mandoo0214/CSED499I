import os
import torch

print(os.path.abspath('pretrained_models/ipnet'))
print(os.path.exists('pretrained_models/ipnet'))

ckpt = torch.load('pretrained_models/ipnet', map_location='cpu')
print(ckpt.keys())
print(list(ckpt['state_dict'].keys())[:10])