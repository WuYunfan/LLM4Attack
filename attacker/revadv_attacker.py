import torch
from torch.optim import Adam, SGD
import numpy as np
from utils import mse_loss, ce_loss, vprint
import torch.nn.functional as F
import higher
import time
from attacker.basic_attacker import BasicAttacker
import gc
from model import get_model
from trainer import get_trainer


def retrain_surrogate(self):
    self.surrogate_model.initial_embeddings()
    self.surrogate_trainer.initialize_optimizer()
    self.surrogate_trainer.best_ndcg = -np.inf
    self.surrogate_trainer.merge_fake_tensor(self.fake_tensor.detach())
    self.surrogate_trainer.train(verbose=False, save=False)

    self.surrogate_trainer.merge_fake_tensor(self.fake_tensor)
    with higher.innerloop_ctx(self.surrogate_model, self.surrogate_trainer.opt) as (fmodel, diffopt):
        fmodel.train()
        for _ in range(self.unroll_steps):
            for users in self.surrogate_trainer.train_user_loader:
                users = users[0]
                if self.save_memory_mode:
                    users = torch.arange(self.n_users, self.n_users + self.n_fakes, dtype=torch.int64,
                                         device=self.device)
                scores, l2_norm_sq = fmodel.forward(users)
                profiles = self.surrogate_trainer.merged_data_tensor[users, :]
                rec_loss = self.surrogate_trainer.loss(profiles, scores, self.surrogate_trainer.weight)
                loss = rec_loss + self.surrogate_trainer.l2_reg * l2_norm_sq.mean()
                diffopt.step(loss)
                if self.save_memory_mode:
                    break

        fmodel.eval()
        scores = fmodel.predict(self.target_user_tensor)
        _, topk_items = scores.topk(self.topk, dim=1)
        hr = torch.eq(topk_items.unsqueeze(2), self.target_item_tensor.unsqueeze(0).unsqueeze(0))
        hr = hr.float().sum(dim=1).mean()
        if self.name == 'LegUPAttacker':
            scores = scores * (scores > torch.min(scores[:, self.target_item_tensor], dim=1, keepdim=True).values)
        adv_loss = ce_loss(scores, self.target_item_tensor)
        adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
    return adv_loss.item(), hr.item(), adv_grads


class GradientAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(GradientAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.adv_epochs = attacker_config['adv_epochs']
        self.lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']

        self.surrogate_model_config['n_fakes'] = self.n_fakes
        self.surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        self.surrogate_trainer = get_trainer(self.surrogate_trainer_config, self.surrogate_model)

        self.fake_tensor = self.init_fake_tensor(self.surrogate_trainer.data_tensor)
        self.adv_opt = SGD([self.fake_tensor], lr=self.lr, momentum=self.momentum)

        if attacker_config.get('uplift_ratio', None) is not None:
            data_tensor = self.surrogate_trainer.data_tensor
            y = torch.zeros([self.n_users], device=self.device, dtype=torch.float)
            for item in self.target_items:
                i_users = torch.nonzero(data_tensor[:, item])[:, 0]
                for user in range(self.n_users):
                    y[user] = y[user] + (data_tensor[user, :][None, :] *  data_tensor[i_users, :]).sum()
            self.target_user_tensor = torch.argsort(y, descending=True)[:int(self.n_users * attacker_config['uplift_ratio'])]
        else:
            self.target_user_tensor = torch.arange(self.n_users, dtype=torch.int64, device=self.device)
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)

    def init_fake_tensor(self, data_tensor):
        degree = torch.sum(data_tensor, dim=1)
        qualified_users = data_tensor[degree <= self.n_inters, :]
        sample_idxes = torch.randint(qualified_users.shape[0], size=[self.n_fakes])
        fake_tensor = torch.clone(qualified_users[sample_idxes, :])
        fake_tensor.requires_grad = True
        return fake_tensor

    def project_fake_tensor(self):
        with torch.no_grad():
            _, items = self.fake_tensor.topk(self.n_inters, dim=1)
            self.fake_tensor.zero_()
            self.fake_tensor.data = torch.scatter(self.fake_tensor, 1, items, 1.)

    def retrain_surrogate(self):
        raise NotImplementedError

    def generate_fake_users(self, verbose=True, writer=None):
        max_hr = -np.inf
        for epoch in range(self.adv_epochs):
            start_time = time.time()

            adv_loss, hr, adv_grads = self.retrain_surrogate()
            if hr > max_hr:
                print('Maximal hit ratio, save fake users.')
                for fake_u in range(self.n_fakes):
                    items = torch.where(self.fake_tensor[fake_u, :] > 0.5)[0].cpu().numpy().tolist()
                    self.fake_user_inters[fake_u] = items
                max_hr = hr

            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.adv_opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.adv_opt.step()
            self.project_fake_tensor()

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            vprint('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}%, Time: {:.3f}s'.
                   format(epoch, self.adv_epochs, adv_loss, self.topk, hr * 100., consumed_time), verbose)
            if writer:
                writer.add_scalar('{:s}/Adv_Loss'.format(self.name), adv_loss, epoch)
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), hr, epoch)


class RevAdvAttacker(GradientAttacker):
    def __init__(self, attacker_config):
        super(RevAdvAttacker, self).__init__(attacker_config)
        self.unroll_steps = attacker_config['unroll_steps']
        self.save_memory_mode = attacker_config['save_memory_mode']

    def retrain_surrogate(self):
        return retrain_surrogate(self)
