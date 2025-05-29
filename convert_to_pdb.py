import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from utils import transforms, reconstruct
from utils.reconstruct import MolReconsError

# 파일 경로
pt_path = 'sample_output/result_5.pt'
r = torch.load(pt_path, weights_only = False)
pos_traj = r['pred_ligand_pos_traj']
v_traj = r['pred_ligand_v_traj']

total = 0

# 한 개라도 복원이 성공하는 샘플 찾기
for i in range(len(pos_traj)):
    try:
        pos = pos_traj[i][-1]
        v = v_traj[i][-1]

        atom_type = transforms.get_atomic_number_from_index(v, mode='add_aromatic')
        aromatic = transforms.is_aromatic_from_index(v, mode='add_aromatic')

        mol = reconstruct.reconstruct_from_generated(pos, atom_type, aromatic)

        pdb_path = f'sample_output/valid_sample_{i}.pdb'
        rdmolfiles.MolToPDBFile(mol, pdb_path)
        print(f'** Sample {i} saved to: {pdb_path}')
        break

    except MolReconsError:
        print(f'** Sample {i} failed.')
        total = total + 1

print(f"Total {total} failed, {40-total} succeed.")
