from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config as get_config
import torch
from dataset import get_dataset
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import shutil
import numpy as np


def objective(trial):
    reg_u = trial.suggest_categorical('reg_u', [1.e-4, 1.e-3, 1.e-2])
    alpha = trial.suggest_categorical('alpha', [1.e-2, 1.e-1, 1.])
    s_l2 = trial.suggest_categorical('s_l2', [1.e-3, 1.e-2, 1.e-1])
    s_lr = trial.suggest_categorical('s_lr', [1.e-2, 1.e-1])
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': s_lr, 'l2_reg': s_l2,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': reg_u, 'prob': 0.9, 'kappa': 1.,
                       'step': 1, 'alpha': alpha, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}

    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset, bottom_ratio=0.01)
    attacker_config['target_items'] = target_items
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users(verbose=False)
    recall = attacker.eval(model_config, trainer_config)
    shutil.rmtree('checkpoints')
    return recall


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'dpa2dl-tuning'
    storage = optuna.storages.RDBStorage(url='sqlite:///../{}.db'.format(study_name))
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.BruteForceSampler())

    study.optimize(objective)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    main()