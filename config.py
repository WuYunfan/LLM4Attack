import numpy as np


def get_amazon_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))



def get_amazon_attacker_config():
    gowalla_attacker_config = []

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 0, 'n_inters': 18, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 217, 'n_inters': 18, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 217, 'n_inters': 18, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)