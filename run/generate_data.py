import numpy as np
import torch
from utils import LLMGenerator, init_run, LLMGeneratorOnline
import os
from dataset import output_inters
import time

def sample_pro(popularity, mask):
    validate_popularity = np.maximum(popularity, 0.) * mask
    return validate_popularity / validate_popularity.sum()

def generate_inter_data(path, n_users, n_inters, candidate_size_1, candidate_size_2, train_ratio=0.8):
    n_train_inters = int(n_inters * train_ratio)
    feats_tensor = torch.load(os.path.join(path, 'feats.pt')).to(dtype=torch.float64, device='cuda')

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
    popularity = popularity / popularity.sum() * n_users * n_inters

    # model_id = 'Qwen/Qwen2.5-7B-Instruct'
    # llm_g = LLMGenerator(model_id)
    llm_g = LLMGeneratorOnline()
    generated_train_data = []
    generated_val_data = []
    for user in range(n_users):
        one_user = list()
        mask = np.ones(n_items, dtype=float)
        history = ''
        history_tensor = torch.zeros_like(feats_tensor[0])

        while len(one_user) < n_inters:
            if len(one_user) == 0:
                item = np.random.choice(n_items, p=sample_pro(popularity, mask))
            else:
                candidates = np.random.choice(n_items, p=sample_pro(popularity, mask), size=candidate_size_1)
                scores = torch.matmul(feats_tensor[candidates, :], history_tensor[:, None]).squeeze()
                top_scores, top_indices = torch.topk(scores, k=candidate_size_2)
                candidates = candidates[top_indices.cpu().numpy()]
                candidates_str = '\n'.join([f'{i}: {feats[c]}' for i, c in enumerate(candidates)])
                llm_outputs = llm_g.generate(history, candidates_str, candidate_size_2)
                item = candidates[int(llm_outputs)]

            one_user.append(item)
            mask[item] = 0.
            history = history + '\n' + feats[item]
            history_tensor = history_tensor + feats_tensor[item, :]
            popularity[item] -= 1
        generated_train_data.append(one_user[:n_train_inters])
        generated_val_data.append(one_user[n_train_inters:])
        if user % 1 == 0:
            print(f'Finish generating user {user}, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    generate_path = os.path.join(os.path.dirname(path), 'gen')
    if not os.path.exists(generate_path):
        os.mkdir(generate_path)
    output_inters(os.path.join(generate_path, 'train.txt'), generated_train_data)
    output_inters(os.path.join(generate_path, 'val.txt'), generated_val_data)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    n_users, n_inters = 1000, 18
    candidate_size_1, candidate_size_2 = 1000, 10
    generate_inter_data('data/Amazon/time', n_users, n_inters, candidate_size_1, candidate_size_2)


if __name__ == '__main__':
    main()