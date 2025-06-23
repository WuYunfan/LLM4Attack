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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import openai
import time
import asyncio

api_key = 'sk-oCdPBwCesg9DCYNBA1E39e90BfCb4f1c91B191Ad68FcEf2a'
base_url = 'https://gptgod.cloud/v1/'
openai.proxies={'http://': 'http://10.128.208.12:8888', 'https://':'http://10.128.208.12:8888'}

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


def vprint(content, verbose, end='\n'):
    if verbose:
        print(content, end=end)


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


def kernel_matrix(A, B, h=4):
    D = torch.cdist(A, B, p=2)
    K = torch.exp(- (D * D) / (2 * h * h))
    return K


def kl_estimate(X, Y, k1=50, k2=10):
    normed_X, normed_Y = F.normalize(X, dim=1, p=2), F.normalize(Y, dim=1, p=2)
    K_XX = kernel_matrix(normed_X, normed_X)
    p_hat = K_XX.mean(dim=1)
    K_XY = kernel_matrix(normed_X, normed_Y)
    q_hat = K_XY.topk(k1, dim=1).values
    nearest_indices = torch.stack([torch.randperm(k1, device=q_hat.device)[:k2] for _ in range(q_hat.shape[0])])
    q_hat = torch.gather(q_hat, 1, nearest_indices).mean(dim=1)
    kl = (torch.log(p_hat) -torch.log(q_hat)).mean()
    return kl


class HeaviTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.where(x <= 0., torch.zeros_like(x), torch.ones_like(x))
        return x

    @staticmethod
    def backward(ctx, dy):
        # 0.5 + 0.5 * torch.tanh(x)
        x,  = ctx.saved_tensors
        dtanh = 1 - x.tanh().pow(2)
        return dy * dtanh * 0.5

class LLMGeneratorOnline:
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"x-foo": "true"},
            timeout=40,
            max_retries=0)
        self.model_id = 'gpt-4o-mini'

    async def fetch_completion(self, prompt, candidate_size):
        attempt = 0
        while True:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "system", "content": prompt}],
                )
                index = int(response.choices[0].message.content.strip())
                if not (0 <= index < candidate_size):
                    raise ValueError(f'index {index} out of range')
                return index
            except Exception as e:
                attempt += 1
                print(f"Error: {e}. Retrying... times: {attempt}")
                await asyncio.sleep(1)

    async def call_api(self, prompts, candidate_size):
        tasks = []
        for prompt in prompts:
            tasks.append(self.fetch_completion(prompt, candidate_size))
            await asyncio.sleep(0.05)
        indices = await asyncio.gather(*tasks)
        return indices

    def generate(self, batch_histories, batch_candidate_str, candidate_size):
        prompts = []
        for history, candidates in zip(batch_histories, batch_candidate_str):
            prompt = \
                "You are simulating a user on an online book-selling platform. " \
                "Your task is to select the next book this user is likely to purchase based on its chronological purchasing history. " \
                f"The user's purchase history in sequential order is as follows: {history}\n" \
                f"Below are candidate books the user might purchase next, with their corresponding indexes: \n{candidates}\n" \
                f"You must directly output the integer index of the most likely next purchase, within the range [0, {candidate_size - 1}]. " \
                "**Do not provide any explanations**. " \
                "**Predict even though none perfectly matches.** " \
                "**Consider the possibility of user interest drift, for example, the user may not always read books from the same author or of the same genre.**"
            prompt = \
                "You are simulating a user on an online news platform. " \
                "Your task is to select the next news article this user is likely to click on based on its chronological reading history. " \
                f"The user's reading history in sequential order is as follows: {history}\n" \
                f"Below are candidate news articles the user might click next, with their corresponding indexes: \n{candidates}\n" \ 
                f"You must directly output the integer index of the most likely next click, within the range [0, {candidate_size - 1}]. " \
                "**Do not provide any explanations**. " \
                "**Predict even though none perfectly matches.** " \ 
                "**Consider the possibility of user interest drift, for example, the user may not always read news about the same topic or from the same category.**"
            prompts.append(prompt)
        indices = asyncio.run(self.call_api(prompts, candidate_size))
        print(indices)
        return indices


class LLMEncoderOnline:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"x-foo": "true"},
            timeout=10,
            max_retries=0)
        self.model_id = 'text-embedding-3-small'

    def encode(self, feats):
        prompts = []
        for feat in feats:
            prompt = "As an intelligent book recommender system, your task is to generate a compelling, well-structured, and informative book recommendation summary. " \
                     "You should not only present the key information provided but also supplement it with relevant insights based on your own knowledge. " \
                     "Ensure that the summary is engaging, concise, and appeals to the target audience. " \
                     f"Below is the book's key information: \n{feat}\n"
            prompts.append(prompt)
        success = False
        attempt = 0
        while not success:
            try:
                response = self.client.embeddings.create(
                    model=self.model_id,
                    input=prompts
                )
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error: {e}. Retrying... times: {attempt}")
                time.sleep(1)
        embeddings = [torch.tensor(d.embedding, dtype=torch.float32) for d in response.data]
        return embeddings
