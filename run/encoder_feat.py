import numpy as np
import torch
from utils import LLMEncoder
import os


def encode_feats_2_vectors(path):
    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    llm = LLMEncoder(model_id)
    feats_tensor = []
    with open(os.path.join(path, 'feats.txt'), 'r') as f:
        feat = f.readline().strip()
        while feat:
            p = feat.find(' ')
            feat = feat[p + 1:]
            feats_tensor.append(llm.encode(feat))
            feat = f.readline().strip()
    feats_tensor = torch.stack(feats_tensor, dim=0)
    torch.save(feats_tensor, os.path.join(path, 'feats.pt'))


def main():
    encode_feats_2_vectors('data/Amazon/time')


if __name__ == '__main__':
    main()