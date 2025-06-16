import os.path
import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_amazon_config as get_config
from config import get_amazon_attacker_config as get_attacker_config
import shutil
import numpy as np


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    seed_list = [2024, 42, 0, 131, 1024]

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]
    attacker_config = get_attacker_config()[0]

    for i in range(5):
        set_seed(seed_list[i])
        real_dataset = get_dataset(dataset_config)
        target_items = get_target_items(real_dataset)
        print('Target items of {:d}th run: {:s}'.format(i, str(target_items)))
        attacker_config['target_items'] = target_items

        dataset_config['path'] = os.path.join(os.path.dirname(dataset_config['path']), 'gen')
        generated_dataset = get_dataset(dataset_config)
        attacker = get_attacker(attacker_config, generated_dataset)
        if os.path.exists(log_path + '-' + str(target_items)):
            shutil.rmtree(log_path + '-' + str(target_items))
        writer = SummaryWriter(log_path + '-' + str(target_items))
        attacker.generate_fake_users(writer=writer)

        new_attacker = get_attacker(attacker_config, real_dataset)
        new_attacker.fake_user_inters = attacker.fake_user_inters
        configs = get_config(device)
        for idx, (_, model_config, trainer_config) in enumerate(configs):
            new_attacker.eval(model_config, trainer_config, writer=writer)
            if idx == 0:
                configs[idx + 1][2]['ckpt_path'] = new_attacker.trainer.save_path
        writer.close()
        shutil.rmtree('checkpoints')


if __name__ == '__main__':
    main()
