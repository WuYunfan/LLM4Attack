import numpy as np
import torch
from utils import LLMEncoder, LLMGenerator
import os
from dataset import output_inters

def generate_inter_data(path, n_users, n_inters, popularity_weight):
    feats_tensor = torch.load(os.path.join(path, 'feats.pt')).to(dtype=torch.float64)
    feats_norm = torch.norm(feats_tensor, dim=1, p=2)
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

    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    llm_g = LLMGenerator(model_id)
    llm_e = LLMEncoder(model_id)
    generated_data = []
    for user in range(n_users):
        one_user = set()
        mask = np.zeros(n_items, dtype=float)

        item = np.random.choice(n_items, p=popularity / popularity.sum())
        one_user.add(item)
        mask[item] = -np.inf
        history = feats[item]
        popularity[item] -= 1
        while len(one_user) < n_inters:
            feat = llm_g.generate(history)
            feat_tensor = llm_e.encode(feat).to(dtype=torch.float64, device='cpu')
            g_scores = torch.matmul(feats_tensor, feat_tensor[:, None]).squeeze() / feats_norm
            g_scores = (g_scores - g_scores.min()) / (g_scores.max() - g_scores.min())
            p_scores = (popularity - popularity.min()) / (popularity.max() - popularity.min())
            scores = g_scores + popularity_weight * p_scores + mask
            item = scores.argmax().item()

            one_user.add(item)
            mask[item] = -np.inf
            history = history + '\n' + feats[item]
            popularity[item] -= 1
        generated_data.append(one_user)
        if user % 10 == 0:
            print(f'Finish generating user {user}.')
    output_inters(os.path.join(path, 'gen_data.txt'), generated_data)


def main():
    popularity_weight = 0.5
    n_users, n_inters = 10000, 20
    generate_inter_data('data/Amazon/time', n_users, n_inters, popularity_weight)


if __name__ == '__main__':
    main()