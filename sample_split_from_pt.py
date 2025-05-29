import torch
import os

src_path = 'output_pdb/sample_100.pt'
dst_path = 'output_pdb/sampled_100_split'
os.makedirs(dst_path, exist_ok=True)

# 파일 로드
data = torch.load(src_path, weights_only=False)

n_samples = len(data['pred_ligand_pos'])
print(f"Total sample: {n_samples}")

for i in range(n_samples):
    out = {
        'data': data['data'], # 동일한 타겟 단백질에 대해 생성되었으므로
        'pred_ligand_pos': data['pred_ligand_pos'][i],
        'pred_ligand_v': data['pred_ligand_v'][i],
        'pred_ligand_pos_traj': data['pred_ligand_pos_traj'][i],
        'pred_ligand_v_traj': data['pred_ligand_v_traj'][i],
    }

    torch.save(out, os.path.join(dst_path, f'result_{i:04d}.pt'))

print("Sample split complete.")
