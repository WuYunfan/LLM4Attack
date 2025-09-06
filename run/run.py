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

"""
target_items_lists = [[10555, 5354, 9431, 9515, 10259], [6600, 10590, 5, 9476, 208],
                      [6597, 10590, 2260, 9342, 927], [6177, 6622, 6408, 2289, 968],
                      [9476, 9461, 10582, 5, 10601]]       
target_items_lists = [[1197, 2459, 1634, 1567, 1329], [2310, 2516, 2744, 2662, 2811],
                      [2243, 2393, 2240, 2349, 1092], [2811, 2840, 2177, 2791, 2464],
                      [2662, 2706, 2177, 2811, 735]]            
target_items_lists = [[8931, 17840, 10625, 15363, 7210], [15489, 2925, 11902, 4725, 11640],
                      [5947, 11938, 17670, 5949, 11817], [17565, 2719, 11289, 11689, 17363],
                      [11347, 10586, 7616, 17654, 3207]]                                            
"""

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
