import argparse
import os
import sys
import shutil
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 밑에 있는 utils를 import하지 못해 작성

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from graphbap.bapnet import BAPNet


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_pdb')
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False) # 공식 pretrained_model을 받은 것이므로 torch.load()의 보안 설정 해제
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)

    net_cond = BAPNet(ckpt_path='pretrained_models/ipnet')

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        net_cond = net_cond
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
