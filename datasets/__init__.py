import torch
from torch.utils.data import Subset # 데이터셋에서 원하는 인덱스만 뽑아내어 필요한 데이터만 모인 subset을 만들도록 해줌
from .pl_pair_dataset import PocketLigandPairDataset # 지금 같은 폴더에 있는 pi_pair_dataset.py 파일에서 "PocketLigandPairDataset"이라는 클래스를 갖고 옴


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config: # 'split'이라는 옵션이 있으면 데이터를 나눔
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
