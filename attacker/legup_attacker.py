import torch.nn as nn
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import numpy as np
from model import init_one_layer
import torch.nn.functional as F
from utils import HeaviTanh, AverageMeter, vprint
from attacker import BasicAttacker
import torch
from model import get_model
from trainer import get_trainer
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from attacker.revadv_attacker import retrain_surrogate
import time


class DiscreteAutoEncoder(nn.Module):
    def __init__(self, layer_sizes, device):
        super(DiscreteAutoEncoder, self).__init__()
        self.e_layer_sizes = layer_sizes
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()

        self.encoder_layers = []
        self.decoder_layers = []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = init_one_layer(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = init_one_layer(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.layers = self.encoder_layers + self.decoder_layers
        self.device = device
        self.to(device=self.device)

    def forward(self, profiles, discrete=True):
        x = F.normalize(profiles, p=2, dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x)
            elif discrete:
                x = HeaviTanh.apply(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, layer_sizes, device):
        super(Discriminator, self).__init__()
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(init_one_layer(layer_sizes[i - 1], layer_sizes[i]))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device=self.device)

    def forward(self, x):
        return self.net(x).squeeze()


class LegUPAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(LegUPAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.n_epochs = attacker_config['n_epochs']
        self.n_pretrain_g_epochs = attacker_config['n_pretrain_g_epochs']
        self.n_pretrain_d_epochs = attacker_config['n_pretrain_d_epochs']
        self.n_g_steps = attacker_config['n_g_steps']
        self.n_d_steps = attacker_config['n_d_steps']
        self.n_attack_steps = attacker_config['n_attack_steps']
        self.unroll_steps = attacker_config['unroll_steps']
        self.save_memory_mode = attacker_config['save_memory_mode']
        self.lr_g = attacker_config['lr_g']
        self.lr_d = attacker_config['lr_d']
        self.reconstruct_weight = attacker_config['reconstruct_weight']

        self.surrogate_model_config['n_fakes'] = self.n_fakes
        self.surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        self.surrogate_trainer = get_trainer(self.surrogate_trainer_config, self.surrogate_model)
        self.g = DiscreteAutoEncoder([self.n_items] + attacker_config['g_layer_sizes'], self.device)
        self.d = Discriminator([self.n_items] + attacker_config['d_layer_sizes'], self.device)
        self.g_opt = Adam(self.g.parameters(), lr=self.lr_g)
        self.d_opt = Adam(self.d.parameters(), lr=self.lr_d)

        self.data_tensor = self.surrogate_trainer.data_tensor
        self.template_indices = torch.randperm(self.n_users)[:self.n_fakes]
        self.fake_tensor = None

        self.target_user_tensor = torch.arange(self.n_users, dtype=torch.int64, device=self.device)
        self.real_user_loader = DataLoader(TensorDataset(self.target_user_tensor),
                                           batch_size=self.n_fakes, shuffle=True)
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)


    def train_d(self, verbose=True, writer=None):
        self.d.train()
        with torch.no_grad():
            self.g.eval()
            fake_tensor = self.g(self.data_tensor[self.template_indices])

        losses = AverageMeter()
        real_corrects = AverageMeter()
        fake_corrects = AverageMeter()
        for real_user_indices in self.real_user_loader:
            real_tensor = self.data_tensor[real_user_indices]
            d_real = self.d(real_tensor)
            real_correct_ratio = d_real.gt(0.5).float().mean()
            d_fake = self.d(fake_tensor)
            fake_correct_ratio = d_fake.le(0.5).float().mean()
            real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
            fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            d_loss = (real_loss + fake_loss) / 2.
            self.d_opt.zero_grad()
            d_loss.backward()
            self.d_opt.step()
            losses.update(d_loss.item(), d_real.shape[0])
            real_corrects.update(real_correct_ratio.item(), d_real.shape[0])
            fake_corrects.update(fake_correct_ratio.item(), d_fake.shape[0])
        vprint(f'Discriminator Loss: {losses.avg:.6f}, Real Correct Ratio: {real_corrects.avg:.4f}, '
               f'Fake Correct Ratio: {fake_corrects.avg:.4f}', verbose)
        if writer:
            writer.add_scalar('{:s}/Discriminator_Loss'.format(self.name), losses.avg)
            writer.add_scalar('{:s}/Real_Correct'.format(self.name), real_corrects.avg)
            writer.add_scalar('{:s}/Fake_Correct'.format(self.name), fake_corrects.avg)

    def train_g(self, stealth=False, attack=False, reconstruct=False, verbose=True, writer=None):
        self.g.train()
        fake_tensor = self.g(self.data_tensor[self.template_indices])
        self.g_opt.zero_grad()
        if stealth:
            self.d.eval()
            d_fake = self.d(fake_tensor)
            stealth_ratio = d_fake.gt(0.5).float().mean()
            stealth_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            stealth_loss.backward()
            vprint(f'Stealth Loss: {stealth_loss:.6f}, Stealth Ratio: {stealth_ratio:.4f}', verbose)
            if writer:
                writer.add_scalar('{:s}/Stealth_Loss'.format(self.name), stealth_loss.item())
                writer.add_scalar('{:s}/Stealth_Ratio'.format(self.name), stealth_ratio.item())

        if attack:
            self.fake_tensor = fake_tensor
            adv_loss, hr, adv_grads = retrain_surrogate(self)
            params = [p for p in self.g.parameters() if p.requires_grad]
            grads = torch.autograd.grad(fake_tensor, params, adv_grads)
            for p, g in zip(params, grads):
                    p.grad = g
            vprint(f'Attack Loss: {adv_loss:.6f}, Hit Ratio: {hr * 100:.4f}%', verbose)
            if writer:
                writer.add_scalar('{:s}/Attack_Loss'.format(self.name), adv_loss)
                writer.add_scalar('{:s}/Hit_Ratio'.format(self.name), hr)

        if reconstruct:
            weight = torch.ones_like(self.data_tensor[self.template_indices])
            weight[self.data_tensor[self.template_indices] > 0.5] = self.reconstruct_weight
            bce_loss = F.binary_cross_entropy(fake_tensor, self.data_tensor[self.template_indices], weight)
            bce_loss.backward()
            vprint(f'Reconstruct BCE Loss: {bce_loss.item():.6f}', verbose)
            if writer:
                writer.add_scalar('{:s}/BCE_Loss'.format(self.name), bce_loss.item())

        self.g_opt.step()

    def train(self, verbose=True, writer=None):
        start_time = time.time()
        print('pretrain G......')
        for epoch in range(self.n_pretrain_g_epochs):
            vprint(f'Pretrain Generator Epoch: {epoch}, ', verbose, end='')
            self.train_g(reconstruct=True, verbose=verbose, writer=writer)

        print('pretrain D......')
        for epoch in range(self.n_pretrain_d_epochs):
            vprint(f'Pretrain Discriminator Epoch: {epoch}, ', verbose, end='')
            self.train_d(verbose=verbose, writer=writer)
        print('======================pretrain end======================\n')
        consumed_time = time.time() - start_time
        self.consumed_time += consumed_time

        for epoch in range(self.n_epochs):
            start_time = time.time()
            vprint(f'==============epoch {epoch}===============', verbose)
            for step_d in range(self.n_d_steps):
                self.train_d(verbose=verbose, writer=writer)

            self.g_opt = Adam(self.g.parameters(), lr=self.lr_g)
            for step_g in range(self.n_g_steps):
                self.train_g(stealth=True, verbose=verbose, writer=writer)

            self.g_opt = Adam(self.g.parameters(), lr=self.lr_g)
            for step_attack in range(self.n_attack_steps):
                vprint(f'==============Step Attack {step_attack}===============', verbose)
                self.train_g(attack=True, verbose=verbose, writer=writer)

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time

    def generate_fake_users(self, verbose=True, writer=None):
        self.train(verbose=verbose, writer=writer)
        with torch.no_grad():
            self.g.eval()
            fake_tensor = self.g(self.data_tensor[self.template_indices], discrete=False)
            filler_items = fake_tensor.topk(self.n_inters).indices
        self.fake_user_inters = [filler_items[u_idx].cpu().numpy().tolist() for u_idx in range(self.n_fakes)]
