import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from utils import transforms, reconstruct
from utils.reconstruct import MolReconsError

# 파일 경로
pt_path = 'output_pdb/sample_100.pt'
r = torch.load(pt_path, weights_only = False)
pos_traj = r['pred_ligand_pos_traj']
v_traj = r['pred_ligand_v_traj']

# 한 개라도 복원이 성공하는 샘플 찾기
for i in range(len(pos_traj)):
    try:
        pos = pos_traj[i][-1]  # 마지막 timestep
        v = v_traj[i][-1]

        atom_type = transforms.get_atomic_number_from_index(v, mode='add_aromatic')
        aromatic = transforms.is_aromatic_from_index(v, mode='add_aromatic')

        mol = reconstruct.reconstruct_from_generated(pos, atom_type, aromatic)

        pdb_path = f'output_pdb/valid_sample_{i}.pdb'
        rdmolfiles.MolToPDBFile(mol, pdb_path)
        print(f'✅ 샘플 {i} 저장 완료: {pdb_path}')
        break

    except MolReconsError:
        print(f'❌ 샘플 {i} 복원 실패')
