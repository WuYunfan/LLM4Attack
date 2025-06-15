import gc
import random
import torch
from torch.cuda import device
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, vprint, get_target_hr, goal_oriented_loss, kl_estimate
import torch.nn.functional as F
import time
import os
import scipy.sparse as sp
from torch.optim import SGD, Adam
import higher


class FEOAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(FEOAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.expected_hr = attacker_config['expected_hr']
        self.step_user = attacker_config['step_user']
        self.n_training_epochs = attacker_config['n_training_epochs']
        self.adv_weight = attacker_config['adv_weight']
        self.kl_weight = attacker_config['kl_weight']
        self.look_ahead_lr = attacker_config['look_ahead_lr']
        self.filler_limit = attacker_config['filler_limit']

        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=True)
        self.validate_topk = attacker_config.get('validate_topk', None)
        if self.validate_topk is not None:
            self.recommendation_lists = []

    def add_filler_items(self, surrogate_model, temp_fake_user_tensor):
        counts = torch.zeros([self.n_items], dtype=torch.int, device=self.device)
        with torch.no_grad():
            scores = surrogate_model.predict(temp_fake_user_tensor)
        for u_idx, f_u in enumerate(temp_fake_user_tensor):
            item_score = scores[u_idx, :]
            if self.validate_topk is not None:
                self.recommendation_lists.append(set(item_score.topk(self.validate_topk).indices.cpu().numpy().tolist()))
            item_score[counts >= self.filler_limit] = 0.
            item_score[self.target_item_tensor] = 0.
            filler_items = item_score.topk(self.n_inters - self.target_items.shape[0]).indices
            counts[filler_items] = counts[filler_items] + 1

            filler_items = filler_items.cpu().numpy().tolist()
            self.fake_user_inters[f_u - self.n_users] = filler_items + self.target_items.tolist()
            self.dataset.train_data[f_u].update(filler_items)
            self.dataset.train_array += [[f_u, item] for item in filler_items]

    def train_fake(self, surrogate_model, surrogate_trainer, temp_fake_user_tensor):
        unroll_train_losses = AverageMeter()
        adv_losses = AverageMeter()
        kl_losses = AverageMeter()
        c_adv_grads, n_batches = torch.zeros_like(surrogate_model.embedding.weight[temp_fake_user_tensor]), 0
        for target_user in self.target_user_loader:
            target_user = target_user[0]
            opt = SGD(surrogate_model.parameters(), lr=self.look_ahead_lr)
            with higher.innerloop_ctx(surrogate_model, opt) as (fmodel, diffopt):
                fmodel.train()
                scores, l2_norm = fmodel.forward(temp_fake_user_tensor)
                score_n, l2_norm_n = scores.mean(dim=1, keepdim=True), l2_norm.mean(dim=1, keepdim=True)
                scores, l2_norm = scores[:, self.target_item_tensor], l2_norm[:, self.target_item_tensor]
                unroll_train_loss = F.softplus(score_n - scores) + surrogate_trainer.l2_reg * (l2_norm + l2_norm_n)
                diffopt.step(unroll_train_loss.sum())

                fmodel.eval()
                scores = fmodel.predict(target_user)
                target_scores = scores[:, self.target_item_tensor]
                top_scores = scores.topk(self.topk, dim=1).values[:, -1:]
                adv_loss =  goal_oriented_loss(target_scores, top_scores, self.expected_hr)
                surrogate_embedding = fmodel.init_fast_params[0]
                adv_grads = torch.autograd.grad(self.adv_weight * adv_loss, surrogate_embedding)[0][temp_fake_user_tensor]
                c_adv_grads, n_batches = c_adv_grads + adv_grads, n_batches + 1

                fake_user_embedding = surrogate_embedding[temp_fake_user_tensor]
                real_user_embedding = surrogate_embedding[:self.n_users]
                kl = kl_estimate(fake_user_embedding, real_user_embedding)
                kl_grads =  torch.autograd.grad(self.kl_weight * kl, surrogate_embedding)[0][temp_fake_user_tensor]
                surrogate_trainer.opt.zero_grad()
                surrogate_model.embedding.weight.grad = torch.zeros_like(surrogate_model.embedding.weight)
                surrogate_model.embedding.weight.grad[temp_fake_user_tensor] = kl_grads
                surrogate_trainer.opt.step()

            unroll_train_losses.update(unroll_train_loss.mean().item())
            adv_losses.update(adv_loss.item(), target_user.shape[0])
            kl_losses.update(kl.item())

        surrogate_trainer.opt.zero_grad()
        surrogate_model.embedding.weight.grad = torch.zeros_like(surrogate_model.embedding.weight)
        surrogate_model.embedding.weight.grad[temp_fake_user_tensor] = c_adv_grads / n_batches
        surrogate_trainer.opt.step()
        return unroll_train_losses.avg, adv_losses.avg, kl_losses.avg

    def retrain_surrogate(self, temp_fake_user_tensor, fake_nums_str, verbose, writer):
        surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)
        for training_epoch in range(self.n_training_epochs):
            start_time = time.time()

            surrogate_model.train()
            t_loss = surrogate_trainer.train_one_epoch(None)
            unroll_train_loss, adv_loss, kl_loss = \
                self.train_fake(surrogate_model, surrogate_trainer, temp_fake_user_tensor)

            target_hr = get_target_hr(surrogate_model, self.target_user_loader, self.target_item_tensor, self.topk)
            consumed_time = time.time() - start_time
            vprint('Training Epoch {:d}/{:d}, Time: {:.3f}s, Train Loss: {:.6f}, Unroll Train Loss: {:.6f}, '
                   'Adv Loss: {:.6f}, KL Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                   format(training_epoch, self.n_training_epochs, consumed_time, t_loss, unroll_train_loss,
                          adv_loss, kl_loss, target_hr * 100.), verbose)
            writer_tag = '{:s}_{:s}'.format(self.name, fake_nums_str)
            if writer:
                writer.add_scalar(writer_tag + '/Hit_Ratio@' + str(self.topk), target_hr, training_epoch)
        self.add_filler_items(surrogate_model, temp_fake_user_tensor)

    def generate_fake_users(self, verbose=True, writer=None):
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step_user, dtype=np.int64)) + [self.n_fakes]
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
            self.dataset.train_array += [[f_u, item] for item in self.target_items for f_u in temp_fake_user_tensor]
            self.dataset.n_users += n_temp_fakes

            self.retrain_surrogate(temp_fake_user_tensor, fake_nums_str, verbose, writer)

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            print('Fake #{:s} has been generated! Time: {:.3f}s'.format(fake_nums_str, consumed_time))
            gc.collect()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
