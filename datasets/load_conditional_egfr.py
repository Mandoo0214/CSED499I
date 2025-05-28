import torch

data = torch.load("egfr_condition.pt", map_location="cpu", weights_only=True)
print("로드 성공:", type(data))

for k, v in data.items():
    print(f"{k}: {type(v)}, shape={getattr(v, 'shape', 'N/A')}")

