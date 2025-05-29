# 이것저것 확인을 위해 필요한 짧은 파이썬 코드들 썼다가 지우는 임시 파일

import torch

data = torch.load('output_pdb/sample_100.pt', weights_only=False)

print(type(data))
for k in data.keys():
    print(k, type(data[k]))
