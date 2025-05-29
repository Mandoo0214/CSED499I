# 이것저것 확인을 위해 필요한 짧은 파이썬 코드들 썼다가 지우는 임시 파일

import torch

# 원본 파일 경로
input_path = 'output_pdb/sample_100.pt'
output_path = 'output_pdb/sample_100_eval.pt'

# 파일 로드
raw_data = torch.load(input_path, weights_only = False)

# 리스트로 되어 있는 위치/벡터 trajectory에서 하나씩 꺼내서 샘플화
pred_pos_traj_list = raw_data['pred_ligand_pos_traj']
pred_v_traj_list = raw_data['pred_ligand_v_traj']
shared_data = raw_data['data']

# 각 샘플을 딕셔너리 형태로 묶어서 리스트에 저장
converted_data = []
for i in range(len(pred_pos_traj_list)):
    sample = {
        'data': shared_data,
        'pred_ligand_pos_traj': pred_pos_traj_list[i],
        'pred_ligand_v_traj': pred_v_traj_list[i],
    }
    converted_data.append(sample)

# 새 파일로 저장
torch.save(converted_data, output_path)
print(f"변환 완료! {len(converted_data)}개의 샘플이 {output_path}에 저장됨.")
