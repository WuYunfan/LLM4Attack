import numpy as np
import torch
from utils import LLMEncoderOnline
import os
import time


def encode_feats_2_vectors(path, batch_size=512):
    llm = LLMEncoderOnline()
    feats_tensor = []
    with open(os.path.join(path, 'feats.txt'), 'r') as f:
        item = 0
        feats = []
        feat = f.readline().strip()
        while feat:
            p = feat.find(' ')
            feat = feat[p + 1:]
            feats.append(feat)
            feat = f.readline().strip()
            if len(feats) == batch_size or len(feat) == 0:
                tensors = llm.encode(feats)
                feats_tensor.extend(tensors)
                feats = []
                print(f'Finish encode item {item}, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')
            item += 1
    feats_tensor = torch.stack(feats_tensor, dim=0)
    torch.save(feats_tensor, os.path.join(path, 'feats.pt'))


def main():
    encode_feats_2_vectors('data/Amazon/time')


if __name__ == '__main__':
    main()