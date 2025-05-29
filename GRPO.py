import os
import sys

# 다른 디렉토리에 위치한 모듈을 찾지 못하는 문제를 해결하기 위해 삽입
# 현재 이 파일이 위치한 디렉토리가 루트이므로, 현재 디렉토리 위치에서 이 파이썬 파일이 실행되도록 강제하면 됨
# 파이썬 파일 실행 위치부터 강제 설정한 뒤, 아래 from ... import 수행
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)

import torch
import argparse
import numpy as np

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.evaluation import atom_num
from graphbap.bapnet import BAPNet

from rdkit import Chem
from rdkit.Chem import QED

from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean

from tqdm import tqdm


# 받아온 디렉토리 안에 있는 pt 파일을 찾아서 파일의 내용을 리스트로 반환
# pt 파일은 하나만 있다고 가정 + 파일 이름을 알고 있다고 가정
def load_samples(sample_dir, fileName):
    pt_path = os.path.join(sample_dir, fileName)

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{fileName} not found.")

    pt_file = torch.load(pt_path, weights_only = False)        
    
    return pt_file


# 각 샘플별 보상 계산 함수
# 메인에서 받은 옵션 및 가중치를 이용하여, 각 분자별 지표와 가중치를 곱해 최종 보상 값을 계산함
def compute_rewards(samples, reward_config):
    rewards = []

    for sample in samples:
        sample_rewards = []

        for mol in sample['all_results']:
            reward = 0.0

            if 'QED' in reward_config:
                reward += reward_config['qed'] * mol['chem_results']['qed']

            if 'SA' in reward_config:
                reward += reward_config['sa'] * mol['chem_results']['sa']

            if 'Vina' in reward_config and mol.get('vina'):
                vina_affinity = mol['vina']['score_only'][0]['affinity']
                reward += reward_config['vina'] * vina_affinity

            sample_rewards.append(reward)

        rewards.append(sample_rewards)

    return rewards


# top k 샘플에 대한 보상 값 평균 계산용 함수
def compute_avg(top_samples, reward_config):
    rewards = []

    for sample in top_samples:
        reward = 0.0

        if 'QED' in reward_config:
            reward += reward_config['qed'] * sample['chem_results']['qed']

        if 'SA' in reward_config:
            reward += reward_config['sa'] * sample['chem_results']['sa']

        if 'Vina' in reward_config:
            reward += reward_config['vina'] * sample['vina']['score_only'][0]['affinity']

        rewards.append(reward)
    
    avg_reward = np.mean(rewards)

    return avg_reward


# top k sample의 reward 평균값을 받아 그 이상의 품질을 갖는 분자를 생성하도록 함
def sample_new_ligands(model, num_samples, reward_avg):
    # 보상 평균에 따라 diffusion time step을 조절하도록 함
    reward_guidance = 1.0 - reward_avg
    adjusted_steps = int(config.sample.num_steps * reward_guidance)

    print(f"Adjusted sampling steps: {adjusted_steps}")

    new_samples = []

    for data_id in range(args.start_index, args.end_index + 1):
        data = test_set[data_id]

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=adjusted_steps,
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

        new_samples.append(result)

        logger.info(f'Sample done for data_id {data_id}')
        print(f'Sampled data_id: {data_id}')

        # 결과를 파일로 저장
        result_path = os.path.join(args.result_path, f'result_{data_id}_grpo.pt')
        torch.save(result, result_path)

    return len(new_samples)


# 실행
if __name__ == '__main__':
    # GRPO에서 보상으로 적용할 지표를 선택하도록 옵션을 받아 parsing 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--qed', type = eval, default = True)
    parser.add_argument('--sa', type = eval, default = True)
    parser.add_argument('--vina', type = eval, default = True)
    parser.add_argument('--q_weight', type = float, default = 1.0)
    parser.add_argument('--s_weight', type = float, default = -0.1)
    parser.add_argument('--v_weight', type = float, default = -0.01)

    parser.add_argument('--config', type = str, default = './configs/sampling.yml')
    parser.add_argument('--train_config', type = str, default = './configs/training.yml')
    parser.add_argument('--sample_num', type = int, default = 20)
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--result_path', type = str, default = './outputs_grpo')
    parser.add_argument('--start_index', type = int, default = 0)
    parser.add_argument('--end_index', type = int, default = 99)

    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    reward_config = {}

    if args.qed:
        reward_config['QED'] = args.q_weight
    if args.sa:
        reward_config['SA'] = args.s_weight
    if args.vina:
        reward_config['Vina'] = args.v_weight

    print("Chosen options and according options: ")
    for metric, weight in reward_config.items():
        print(f"  - {metric}: {weight}")

    # pt 파일 로드
    sample_dir = './sample_output' # metric를 저장한 pt 파일이 들어있는 디렉토리, 변경에 용이하도록 변수로 관리
    fileName = 'result_5.pt' # metric이 저장된 pt 파일 이름, 변경에 용이하도록 변수로 관리

    samples = load_samples(sample_dir, fileName)

    # 보상 계산
    rewards = compute_rewards(samples, reward_config)

    # 분자들 중 위에서 구한 보상 값이 가장 좋은 상위 10개 분자를 골라 냄
    # rewards는 리스트이므로 torch.topk()를 직접 사용하지 못함
    topk = 10 # 변수로 관리하여 변경이 용이하도록 함
    top_samples = [samples[i] for i in rewards.argsort()[-topk:]] # 오름차순이므로 뒤 topk개를 잘라내야 함
    print(f"Top {topk} samples selected.")

    # 뽑아낸 top 10 샘플들이 갖는 보상 값의 평균을 계산
    reward_avg = compute_avg(top_samples, reward_config)
    print(f"Reward value of top {topk} samples: {reward_avg}")

    # 모델을 로드하고 실제 보상값을 반영하여 새로운 샘플 생성
    # config 로드
    config = misc.load_config(args.config)
    train_config = misc.load_config(args.train_config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # checkpoint 로드
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    # 이 파일에서는 Crossdock dataset을 활용한다고 가정
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # dataset 로드
    dataset, subsets = get_dataset(
        config=train_config.data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # 모델 로드
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    net_cond = BAPNet(ckpt_path=train_config.net_cond.ckpt_path, hidden_nf=train_config.net_cond.hidden_dim).to(args.device)

    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # 보상값 반영한 샘플링 수행 및 샘플 파일 저장
    new_samples = sample_new_ligands(model, args.sample_num, reward_avg)

    # 결과 출력
    print(f"Total {new_samples} samples are successfully generated.")
    print(f"New samples saved to: {args.result_path}")

