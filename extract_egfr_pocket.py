# extract_egfr_pocket.py

from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import torch
import argparse
import os

# 리간드(AQ4) 중심 6.0A 내의 단백질 원자 추출
def extract_pocket_atoms(pdb_path, ligand_resname='AQ4', radius=6.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    model = structure[0]
    atoms = list(model.get_atoms())

    # 리간드 원자들
    ligand_atoms = [a for a in atoms if a.get_parent().get_resname().strip() == ligand_resname]
    if not ligand_atoms:
        raise ValueError(f"ligand {ligand_resname} is not in pdb file")

    ligand_center = np.mean([atom.get_coord() for atom in ligand_atoms], axis=0)

    # 이웃 탐색
    ns = NeighborSearch(atoms)
    pocket_atoms = ns.search(ligand_center, radius, level='A')

    # 단백질 원자만 필터링
    pocket_atoms = [
        atom for atom in pocket_atoms
        if atom.get_parent().get_id()[0] == ' '  # ' '는 단백질
    ]

    pos = np.array([a.get_coord() for a in pocket_atoms])
    types = [a.element for a in pocket_atoms]

    return {
        'protein_pos': pos,
        'protein_type': types
    }

# 추출한 단백질 원자들을 pt 파일로 저장
def save_protein_condition(result_dict, output_path):
    vocab = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'H': 4}
    type_idx = [vocab.get(t, -1) for t in result_dict['protein_type']]

    pos_tensor = torch.tensor(result_dict['protein_pos'], dtype=torch.float32)
    type_tensor = torch.tensor(type_idx, dtype=torch.long)

    data = {
        'protein_pos': pos_tensor,
        'protein_type': type_tensor
    }
    torch.save(data, output_path)
    print(f"Successfully saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="extract pocket and save it to pt file")
    parser.add_argument('--pdb', type=str, required=True, help='input pdb file path')
    parser.add_argument('--ligand', type=str, default='AQ4', help='ligand code')
    parser.add_argument('--radius', type=float, default=6.0, help='extraction radius')
    parser.add_argument('--out', type=str, required=True, help='output path')

    args = parser.parse_args()

    result = extract_pocket_atoms(args.pdb, ligand_resname=args.ligand, radius=args.radius)
    save_protein_condition(result, args.out)

if __name__ == "__main__":
    main()
