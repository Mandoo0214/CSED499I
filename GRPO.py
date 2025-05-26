from diffusion_model import IPDiffSampler  # 너가 분석한 모델에서 샘플러
from reward_metrics import compute_reward  # 네가 만든 QED/affinity 평가 함수

for step in range(num_iterations):
    molecules = IPDiffSampler.sample(batch_size)

    # reward 평가
    rewards = [compute_reward(mol) for mol in molecules]

    # policy gradient 기반으로 loss 계산 (GRPO 핵심)
    log_probs = IPDiffSampler.get_log_probs(molecules)
    loss = -torch.mean(torch.tensor(rewards) * log_probs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
