import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from graphbap.bapnet import BAPNet

from sample_split import sample_diffusion_ligand


def prepare_single_sample(condition_pt_path, device):
    cond = torch.load(condition_pt_path, map_location=device)
    pos = cond['protein_pos']
    types = cond['protein_type']
    num_atoms = len(types)
    num_ligand_atoms = 10

    data_obj = Data()

    data_obj.protein_pos = pos
    data_obj.protein_element = types
    data_obj.protein_type = types
    
    # dummy data
    data_obj.protein_atom_to_aa_type = torch.zeros(num_atoms, dtype=torch.long)
    data_obj.protein_atom_to_aa = torch.arange(num_atoms)
    data_obj.protein_chain_id = torch.zeros(num_atoms, dtype=torch.long)
    data_obj.protein_is_backbone = torch.ones(num_atoms, dtype=torch.bool)

    data_obj.ligand_mol_type = 'ligand'
    data_obj.ligand_pos = torch.zeros((1, 3), dtype=torch.float32).to(device)
    data_obj.ligand_element = torch.tensor([6], dtype=torch.long).to(device)
    data_obj.ligand_bond_index = torch.empty((2, 0), dtype=torch.long).to(device)
    data_obj.ligand_bond_type = torch.empty((0,), dtype=torch.long).to(device)

    data_obj.ligand_hybridization = torch.tensor([2], dtype=torch.long).to(device)
    data_obj.ligand_valence = torch.tensor([4], dtype=torch.long).to(device)
    data_obj.ligand_formal_charge = torch.tensor([0], dtype=torch.long).to(device)
    data_obj.ligand_is_aromatic = torch.tensor([False], dtype=torch.bool).to(device)
    data_obj.ligand_chiral_tag = torch.tensor([0], dtype=torch.long).to(device)
    data_obj.ligand_atom_map = torch.tensor([0], dtype=torch.long).to(device)
    data_obj.ligand_ring_atom = torch.tensor([False], dtype=torch.bool).to(device)
    data_obj.ligand_num_hs = torch.tensor([3], dtype=torch.long).to(device)
    data_obj.ligand_mass = torch.tensor([12.01], dtype=torch.float32).to(device)  # carbon
    data_obj.ligand_is_in_ring = torch.tensor([False], dtype=torch.bool).to(device)

    data_obj.ligand_atom_feature = torch.zeros((num_ligand_atoms, 15), dtype=torch.float32).to(device)

    data_obj.y = torch.tensor(0)

    return data_obj


def sample_from_condition(config, train_config, model_ckpt_path, condition_pt_path, save_path,
                           num_samples=10, batch_size=5, device='cuda:0'):

    # Seed & logger
    misc.seed_all(config.sample.seed)
    os.makedirs(save_path, exist_ok=True)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(train_config.data.transform.ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond()
    ])

    # Load model
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(device)
    ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    net_cond = BAPNet(ckpt_path=train_config.net_cond.ckpt_path,
                      hidden_nf=train_config.net_cond.hidden_dim).to(device)

    # Prepare condition data
    data = prepare_single_sample(condition_pt_path, device)
    data = transform(data)
    data = data.to(device)

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, _, _, time_list = sample_diffusion_ligand(
        model, data, num_samples,
        batch_size=batch_size, device=device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        net_cond=net_cond,
        cond_dim=train_config.model.cond_dim
    )

    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }

    torch.save(result, os.path.join(save_path, f'egfr_conditional_result.pt'))
    print(f"\n Conditional sampling successfully done. Total {num_samples} atoms → {save_path}/egfr_conditional_result.pt")


# 예시 실행:
if __name__ == '__main__':
    from utils.misc import load_config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='sampling.yml path')
    parser.add_argument('--train_config', type=str, required=True, help='training.yml path')
    parser.add_argument('--condition', type=str, required=True, help='EGFR pocket file(.pt)')
    parser.add_argument('--ckpt', type=str, required=True, help='pre-trained model')
    parser.add_argument('--save_path', type=str, default='./egfr_samples')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = load_config(args.train_config)

    sample_from_condition(config, train_config, args.ckpt, args.condition,
                          args.save_path, args.num_samples, args.batch_size, args.device)