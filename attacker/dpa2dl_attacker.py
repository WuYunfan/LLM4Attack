import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, topk_loss, vprint, get_target_hr
import torch.nn.functional as F
import time
import os


class DPA2DLAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(DPA2DLAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.reg_u = attacker_config['reg_u']
        self.prob = attacker_config['prob']
        self.kappa = torch.tensor(attacker_config['kappa'], dtype=torch.float32, device=self.device)
        self.step = attacker_config['step']
        self.alpha = attacker_config['alpha']
        self.n_rounds = attacker_config['n_rounds']

        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        non_target_items = list(set(range(self.n_items)) - set(self.target_items.tolist()))
        self.non_target_item_tensor = torch.tensor(non_target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=True)

    def poison_train(self, surrogate_model, surrogate_trainer, temp_fake_user_tensor):
        losses = AverageMeter()
        for users in self.target_user_loader:
            users = users[0]
            scores = surrogate_model.predict(users)
            loss = self.alpha * topk_loss(scores, self.target_item_tensor, self.topk, self.kappa)
            surrogate_trainer.opt.zero_grad()
            loss.backward()
            surrogate_trainer.opt.step()
            losses.update(loss.item(), users.shape[0] * self.target_items.shape[0])

        scores = surrogate_model.predict(temp_fake_user_tensor)
        scores = scores[:, self.non_target_item_tensor]
        loss = self.reg_u * (torch.sigmoid(scores) ** 2).mean()
        surrogate_trainer.opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        losses.update(loss.item(), temp_fake_user_tensor.shape[0] * self.non_target_item_tensor.shape[0])
        return losses.avg

    def choose_filler_items(self, surrogate_model, temp_fake_user_tensor, prob):
        surrogate_model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(surrogate_model.predict(temp_fake_user_tensor))
        for u_idx, f_u in enumerate(temp_fake_user_tensor):
            item_score = scores[u_idx, :] * prob
            item_score[self.target_item_tensor] = 0.
            filler_items = item_score.topk(self.n_inters - self.target_items.shape[0]).indices
            prob[filler_items] *= self.prob
            if (prob < 1.0).all():
                prob[:] = 1.

            filler_items = filler_items.cpu().numpy().tolist()
            self.fake_user_inters[f_u - self.n_users] = filler_items + self.target_items.tolist()
            self.dataset.train_data[f_u].update(filler_items)
            self.dataset.train_array += [[f_u, item] for item in filler_items]

    def generate_fake_users(self, verbose=True, writer=None):
        prob = torch.ones(self.n_items, dtype=torch.float32, device=self.device)
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating fake #{:s} !'.format(fake_nums_str))
            temp_fake_user_tensor = np.arange(fake_user_end_indices[i_step - 1],
                                              fake_user_end_indices[i_step]) + self.n_users
            temp_fake_user_tensor = torch.tensor(temp_fake_user_tensor, dtype=torch.int64, device=self.device)
            n_temp_fakes = temp_fake_user_tensor.shape[0]

            self.dataset.train_data += [set(self.target_items) for _ in range(n_temp_fakes)]
            self.dataset.val_data += [set() for _ in range(n_temp_fakes)]
            self.dataset.train_array += [[fake_u, item] for item in self.target_items for fake_u in
                                         temp_fake_user_tensor]
            self.dataset.n_users += n_temp_fakes

            surrogate_model = get_model(self.surrogate_model_config, self.dataset)
            surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
            surrogate_trainer.train(verbose=verbose, save=False)

            target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
            print('Initial target HR: {:.4f}'.format(target_hr))
            for i_round in range(self.n_rounds):
                surrogate_model.train()
                p_loss = self.poison_train(surrogate_model, surrogate_trainer, temp_fake_user_tensor)
                t_loss = surrogate_trainer.train_one_epoch(None)
                target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
                vprint('Round {:d}/{:d}, Poison Loss: {:.6f}, Train Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                       format(i_round, self.n_rounds, p_loss, t_loss, target_hr * 100.), verbose)
                if writer:
                    writer.add_scalar('{:s}_{:s}/Poison_Loss'.format(self.name, fake_nums_str), p_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss'.format(self.name, fake_nums_str), t_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Hit_Ratio@{:d}'.format(self.name, fake_nums_str, self.topk),
                                      target_hr, i_round)
            self.choose_filler_items(surrogate_model, temp_fake_user_tensor, prob)
            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            print('Fake #{:s} has been generated! Time: {:.3f}s'.format(fake_nums_str, consumed_time))
            gc.collect()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
