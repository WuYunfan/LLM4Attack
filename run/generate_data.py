import numpy as np
import torch
from utils import LLMGenerator, init_run
import os
from dataset import output_inters
import time

def sample_pro(popularity, mask):
    validate_popularity = np.maximum(popularity, 0.) * mask
    return validate_popularity / validate_popularity.sum()

def generate_inter_data(path, n_users, n_inters, candidate_size=10):
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
    generated_data = []
    for user in range(n_users):
        one_user = set()
        mask = np.ones(n_items, dtype=float)

        item = np.random.choice(n_items, p=sample_pro(popularity, mask))
        one_user.add(item)
        mask[item] = 0.
        history = feats[item]
        popularity[item] -= 1
        while len(one_user) < n_inters:
            candidates = np.random.choice(n_items, p=sample_pro(popularity, mask), size=candidate_size)
            candidates_str = '\n'.join([f'{i}: {feats[c]}' for i, c in enumerate(candidates)])
            llm_outputs = llm_g.generate(history, candidates_str, candidate_size)
            try:
                item = candidates[int(llm_outputs)]
            except:
                print(llm_outputs)

            one_user.add(item)
            mask[item] = 0.
            history = history + '\n' + feats[item]
            popularity[item] -= 1
        generated_data.append(one_user)
        if user % 1 == 0:
            print(f'Finish generating user {user}, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')
    output_inters(os.path.join(path, 'gen_data.txt'), generated_data)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    n_users, n_inters = 10000, 20
    generate_inter_data('data/Amazon/time', n_users, n_inters)


if __name__ == '__main__':
    main()