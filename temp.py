# 이것저것 확인을 위해 필요한 짧은 파이썬 코드들 썼다가 지우는 임시 파일

import torch

# 경로는 네 파일 위치에 맞게 바꿔줘
pt_path = 'output_pdb/sampled_100_split'

# 파일 불러오기
data = torch.load(pt_path)

# 최상위 키 확인
print("Top-level keys:", data.keys() if isinstance(data, dict) else type(data))

# 내부 구조 확인
if isinstance(data, list):
    print(f"Total samples: {len(data)}")
    print("First item type:", type(data[0]))
    if hasattr(data[0], '__dict__'):
        print("First sample attributes:", data[0].__dict__.keys())
elif isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}")
        if isinstance(v, list):
            print(f" -> list with {len(v)} items, first item type: {type(v[0]) if v else 'empty'}")
        elif isinstance(v, dict):
            print(f" -> dict with keys: {v.keys()}")
else:
    print("Data format not recognized.")

