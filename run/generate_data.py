import numpy as np
import torch
from utils import init_run, LLMGeneratorOnline
import os
from dataset import output_inters
import time

def sample_pro(popularity, mask):
    validate_popularity = np.maximum(popularity, 0.01) * mask
    return validate_popularity / validate_popularity.sum()

def generate_inter_data(path, n_users, n_inters,
                        candidate_size_1=1000, candidate_size_2=10,
                        train_ratio=0.8, batch_size=512):
    n_train_inters = int(n_inters * train_ratio)
    feats_tensor = torch.load(os.path.join(path, 'feats.pt')).to(dtype=torch.float, device='cuda')

    feats = []
    popularity =[]
    with open(os.path.join(path, 'feats.txt'), 'r') as f:
        feat = f.readline().strip()
        while feat:
            p = feat.find(' ')
            popularity.append(int(feat[:p]))
            feat = feat[p + 1:]
            feats.append(feat)
            feat = f.readline().strip()
    n_items = len(popularity)
    popularity = np.array(popularity)

    llm_g = LLMGeneratorOnline()
    generated_train_data = []
    generated_val_data = []
    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        current_batch_size = batch_end - batch_start

        batch_user_data = [list() for _ in range(current_batch_size)]
        batch_masks = np.ones((current_batch_size, n_items), dtype=float)
        batch_histories = ['' for _ in range(current_batch_size)]
        batch_history_tensors = torch.zeros((current_batch_size, feats_tensor.shape[1]), device='cuda', dtype=torch.float)

        n_generated_inters = 0
        while n_generated_inters < n_inters:
            if n_generated_inters != 0:
                batch_candidates = []
                batch_candidates_str = []
                for user_idx in range(current_batch_size):
                    candidates = np.random.choice(n_items, p=sample_pro(popularity, batch_masks[user_idx]),
                                                  size=candidate_size_1)
                    scores = torch.matmul(feats_tensor[candidates, :],
                                          batch_history_tensors[user_idx][:, None]).squeeze()
                    _, top_indices = torch.topk(scores, k=candidate_size_2)
                    candidates = candidates[top_indices.cpu().numpy()]
                    batch_candidates.append(candidates)
                    candidates_str = '\n'.join([f'{i}: {feats[c]}' for i, c in enumerate(candidates)])
                    batch_candidates_str.append(candidates_str)
                indices = llm_g.generate(batch_histories, batch_candidates_str, candidate_size_2)

            for user_idx in range(current_batch_size):
                if n_generated_inters == 0:
                    item = np.random.choice(n_items, p=sample_pro(popularity, batch_masks[user_idx]))
                else:
                    item = batch_candidates[user_idx][indices[user_idx]]
                batch_user_data[user_idx].append(item)
                batch_masks[user_idx][item] = 0.
                batch_histories[user_idx] = batch_histories[user_idx] + '\n' + feats[item]
                batch_history_tensors[user_idx] = batch_history_tensors[user_idx] + feats_tensor[item, :]
            n_generated_inters += 1

        for user_data in batch_user_data:
            generated_train_data.append(user_data[:n_train_inters])
            generated_val_data.append(user_data[n_train_inters:])
        print(f'Finish generating user {batch_end - 1}, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    generate_path = os.path.join(os.path.dirname(path), 'gen')
    if not os.path.exists(generate_path):
        os.mkdir(generate_path)
    output_inters(os.path.join(generate_path, 'train.txt'), generated_train_data)
    output_inters(os.path.join(generate_path, 'val.txt'), generated_val_data)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    n_users, n_inters = 1000, 18
    generate_inter_data('data/Amazon/time', n_users, n_inters)


if __name__ == '__main__':
    main()