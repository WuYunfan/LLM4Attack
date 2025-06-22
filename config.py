import numpy as np


def get_amazon_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/time',
                      'device': device}
    amazon_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.001,
                      'eps': 10., 'adv_reg': 0.01, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/time',
                      'device': device}
    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-3,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-2,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [50], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0., 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048,
                      'test_batch_size': 2048, 'topks': [50]}
    amazon_config.append((dataset_config, model_config, trainer_config))
    return amazon_config


def get_amazon_attacker_config():
    amazon_attacker_config = []

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 0, 'n_inters': 18, 'topk': 50}
    amazon_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 217, 'n_inters': 18, 'topk': 50}
    amazon_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 217, 'n_inters': 18, 'topk': 50}
    amazon_attacker_config.append(attacker_config)
    return amazon_attacker_config