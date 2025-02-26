import numpy as np
import torch
import os
import sys
import scipy.sparse as sp
import torch.nn.functional as F
import dgl
import gc
import random
import types
from functools import partial
from torch.utils.data import Dataset
from dataset import get_negative_items


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, seed):
    set_seed(seed)
    if not os.path.exists(log_path): os.mkdir(log_path)
    f = open(os.path.join(log_path, 'log.txt'), 'w')
    f = Unbuffered(f)
    sys.stderr = f
    sys.stdout = f


def generate_adj_mat(train_array, model):
    train_array = torch.tensor(train_array, dtype=torch.int64, device=model.device)
    users, items = train_array[:, 0], train_array[:, 1]
    row = torch.cat([users, items + model.n_users])
    col = torch.cat([items + model.n_users, users])
    adj_mat = TorchSparseMat(row, col, (model.n_users + model.n_items,
                                        model.n_users + model.n_items), model.device)
    return adj_mat


class TorchSparseMat:
    def __init__(self, row, col, shape, device):
        self.shape = shape
        self.device = device
        self.g = dgl.graph((col, row), num_nodes=max(shape), device=device)

        eps = torch.tensor(1.e-8, dtype=torch.float32, device=device)
        values = torch.ones([self.g.num_edges()], dtype=torch.float32, device=device)
        degree = dgl.ops.gspmm(self.g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
        degree = torch.maximum(degree, eps)
        self.inv_deg = torch.pow(degree, -0.5)

    def spmm(self, r_mat, values=None, norm=None):
        if values is None:
            values = torch.ones([self.g.num_edges()], dtype=torch.float32, device=self.device)
        assert r_mat.shape[0] == self.shape[1]
        padding_tensor = torch.empty([max(self.shape) - r_mat.shape[0], r_mat.shape[1]],
                                     dtype=torch.float32, device=self.device)
        padded_r_mat = torch.cat([r_mat, padding_tensor], dim=0)

        col, row = self.g.edges()
        if norm == 'both':
            values = values * self.inv_deg[row] * self.inv_deg[col]
        if norm == 'right':
            values = values * self.inv_deg[col] * self.inv_deg[col]
        if norm == 'left':
            values = values * self.inv_deg[row] * self.inv_deg[row]
        x = dgl.ops.gspmm(self.g, 'mul', 'sum', lhs_data=padded_r_mat, rhs_data=values)
        return x[:self.shape[0], :]


class AverageMeter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def get_target_items(dataset, bottom_ratio=0.01, num=5):
    data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                             shape=(dataset.n_users, dataset.n_items), dtype=np.float32).tocsr()
    item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
    selected_items = np.argsort(item_degree)[:int(dataset.n_items * bottom_ratio)]
    target_items = np.random.choice(selected_items, size=num, replace=False)
    return target_items


def topk_loss(scores, target_item_tensor, topk, kappa):
    top_scores, _ = scores.topk(topk, dim=1)
    target_scores = scores[:, target_item_tensor]
    loss = F.logsigmoid(top_scores[:, -1:]) - F.logsigmoid(target_scores)
    loss = torch.max(loss, -kappa).mean()
    return loss


def mse_loss(profiles, scores, weight):
    weights = torch.where(profiles > 0.5, weight, 1.)
    loss = weights * (profiles - scores) ** 2
    loss = torch.mean(loss)
    return loss


def ce_loss(scores, target_item_tensor):
    log_probs = F.log_softmax(scores, dim=1)
    return -log_probs[:, target_item_tensor].mean()


def vprint(content, verbose):
    if verbose:
        print(content)


def get_target_hr(surrogate_model, target_user_loader, target_item_tensor, topk):
    surrogate_model.eval()
    with torch.no_grad():
        hrs = AverageMeter()
        for users in target_user_loader:
            users = users[0]
            scores = surrogate_model.predict(users)
            _, topk_items = scores.topk(topk, dim=1)
            hr = torch.eq(topk_items.unsqueeze(2), target_item_tensor.unsqueeze(0).unsqueeze(0))
            hr = hr.float().sum(dim=1).mean()
            hrs.update(hr.item(), users.shape[0])
    return hrs.avg


def goal_oriented_loss(target_scores, top_scores, expected_hr):
    loss = -F.softplus(top_scores.detach() - target_scores)
    n_target_hits = int(expected_hr * loss.shape[0] * loss.shape[1])
    bottom_loss = loss.reshape(-1).topk(n_target_hits).values
    bottom_loss = -bottom_loss
    return bottom_loss.mean()
