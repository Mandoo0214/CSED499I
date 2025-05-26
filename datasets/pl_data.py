import torch
import torch_scatter # element들을 흩뿌리는 역할을 함
import numpy as np
from torch_geometric.data import Data # 그래프 데이터를 표현하는 기본 단위
from torch_geometric.loader import DataLoader

# 각 노드들이 어떤 그래프에 들어가는지를 나타내어 주는 batch를 어떻게 생성하고 관리할지 기준을 제시
# 상호작용을 위해 그래프 노드들을 한번에 묶어 나타낼 때 각 노드들이 속한 그래프를 표현해주는 역할을 함
FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod # 데코레이터, 객체 선언 없이도 사용할 수 있음
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        # item: 텐서에서 int, float 같은 숫자 하나를 뽑아낼 때 사용
        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        # i번째 원자에 대해 edge로 연결된 원자들이 어떤 것들이 있는지 모두 정리함
        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        return instance

    # 그래프의 인덱스가 겹치지 않도록 오프셋을 주는 함수 (자동으로 누적하여 겹치지 않는 오프셋을 생성해줌)
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value) # super: 부모 클래스를 의미함(여기서는 상속받은 클래스인 Data), 해당 부모 클래스에도 동일한 __inc__ 함수가 있으므로 그것을 사용하겠다는 뜻


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)
        # 이것도 부모 클래스(여기서는 DataLoader)의 __init__을 불러서 전달받은 파라미터들을 실제로 대입하도록 함


# 데이터 중에 numpy 타입으로 되어있는 게 있다면 그것을 torch.Tensor로 변환해주는 역할을 함
# pyTorch가 torch.Tensor만 받아 작동할 수 있기 때문
def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    # 각 원자별 그래프의 element 개수가 몇 개인지, ligand_batch를 깨서 새로운 텐서를 얻음
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size # 그래프별 element 개수를 얻은 것을 바탕으로, 누적 원자 개수를 계산하고(cumsum) 이를 바탕으로 각 그래프별 offset index를 구함
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []

    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index] # 특정 원자에 대한 결합만 골라내기

        # offset을 빼 시작 인덱스를 0으로 만들어주기
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]

        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int) # 비어있는 행렬 만듦
        
        # 각 결합의 type을 저장하여 행렬을 만듦
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix) # 만든 행렬을 덧붙임
    return batch_connectivity_matrix
