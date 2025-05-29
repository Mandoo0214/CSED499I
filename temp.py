# 이것저것 확인을 위해 필요한 짧은 파이썬 코드들 썼다가 지우는 임시 파일

import torch
import numpy as np

# 기존 pt 파일 경로
old_path = 'output_pdb/sample_100.pt'
new_path = 'output_pdb/sample_100_eval.pt'

# 기존 파일 로드
r = torch.load(old_path, weights_only = False)

# 리스트 → numpy 배열로 변환
new_r = {
    'data': r['data'],
    'pred_ligand_pos': np.array(r['pred_ligand_pos']),
    'pred_ligand_v': np.array(r['pred_ligand_v']),
    'pred_ligand_pos_traj': np.array(r['pred_ligand_pos_traj']),
    'pred_ligand_v_traj': np.array(r['pred_ligand_v_traj']),
}

# 새로운 파일로 저장
torch.save(new_r, new_path)

print(f'변환 완료! 저장된 파일: {new_path}')
