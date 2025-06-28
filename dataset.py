import os
import numpy as np
from torch.utils.data import Dataset
import random
import sys
import time
import json
import torch
import re
from datetime import datetime
from utils import str_prob


def get_dataset(config):
    config = config.copy()
    dataset = getattr(sys.modules['dataset'], config['name'])
    dataset = dataset(config)
    return dataset


def update_ui_sets(u, i, user_inter_sets, item_inter_sets):
    if u in user_inter_sets:
        user_inter_sets[u].add(i)
    else:
        user_inter_sets[u] = {i}
    if i in item_inter_sets:
        item_inter_sets[i].add(u)
    else:
        item_inter_sets[i] = {u}


def update_user_inter_lists(u, i, t, user_map, item_map, user_inter_lists):
    if u in user_map and i in item_map:
        duplicate = False
        for i_t in user_inter_lists[user_map[u]]:
            if i_t[0] == item_map[i]:
                i_t[1] = min(i_t[1], t)
                duplicate = True
                break
        if not duplicate:
            user_inter_lists[user_map[u]].append([item_map[i], t])


def output_inters(file_path, data, start=0, mode='w'):
    with open(file_path, mode) as f:
        for user in range(len(data)):
            u_items = [str(start + user)] + [str(item) for item in data[user]]
            f.write(' '.join(u_items) + '\n')


def output_feats(file_path, feats):
    with open(file_path, 'w') as f:
        for item in range(len(feats)):
            f.write(feats[item] + '\n')


def get_negative_items(dataset, user, num):
    pos_items = dataset.train_data[user]
    neg_items = np.zeros((num, ), dtype=np.int64)
    for i in range(num):
        item = random.randint(0, dataset.n_items - 1)
        while item in pos_items:
            item = random.randint(0, dataset.n_items - 1)
        neg_items[i] = item
    return neg_items


class BasicDataset(Dataset):
    def __init__(self, dataset_config):
        print(dataset_config)
        self.config = dataset_config
        self.name = dataset_config['name']
        self.min_interactions = dataset_config.get('min_inter')
        self.train_ratio = dataset_config.get('train_ratio')
        self.device = dataset_config['device']
        self.negative_sample_ratio = 1
        self.shuffle = dataset_config.get('shuffle', False)
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.val_data = None
        self.attack_data = None
        self.train_array = None
        self.feats = None
        print('init dataset ' + dataset_config['name'])

    def remove_sparse_ui(self, user_inter_sets, item_inter_sets):
        not_stop = True
        while not_stop:
            not_stop = False
            users = list(user_inter_sets.keys())
            for user in users:
                if len(user_inter_sets[user]) < self.min_interactions:
                    not_stop = True
                    for item in user_inter_sets[user]:
                        item_inter_sets[item].remove(user)
                    user_inter_sets.pop(user)
            items = list(item_inter_sets.keys())
            for item in items:
                if len(item_inter_sets[item]) < self.min_interactions:
                    not_stop = True
                    for user in item_inter_sets[item]:
                        user_inter_sets[user].remove(item)
                    item_inter_sets.pop(item)
        user_map = dict()
        for idx, user in enumerate(user_inter_sets):
            user_map[user] = idx
        item_map = dict()
        for idx, item in enumerate(item_inter_sets):
            item_map[item] = idx
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        return user_map, item_map

    def generate_inters(self, user_inter_lists):
        self.train_data = []
        self.val_data = []
        self.train_array = []
        average_inters = []
        for user in range(self.n_users):
            user_inter_lists[user].sort(key=lambda entry: entry[1])
            if self.shuffle:
                np.random.shuffle(user_inter_lists[user])

            n_inter_items = len(user_inter_lists[user])
            average_inters.append(n_inter_items)
            n_train_items = int(n_inter_items * self.train_ratio)
            self.train_data.append({i_t[0] for i_t in user_inter_lists[user][:n_train_items]})
            self.val_data.append({i_t[0] for i_t in user_inter_lists[user][n_train_items:]})
        average_inters = np.mean(average_inters)
        print('Users {:d}, Items {:d}, Average number of interactions {:.3f}, Total interactions {:.1f}'
              .format(self.n_users, self.n_items, average_inters, average_inters * self.n_users))

    def generate_feats(self, feats, item_map, popularity):
        self.feats = [str(popularity[item]) + ' ' for item in range(self.n_items)]
        for item, feat in feats.items():
            if item in item_map:
                self.feats[item_map[item]] = self.feats[item_map[item]] + feat
        assert all(not s.endswith(' ') for s in self.feats), "all items must have features"

    def __len__(self):
        return len(self.train_array)

    def __getitem__(self, index):
        user = random.randint(0, self.n_users - 1)
        while len(self.train_data[user]) == 0:
            user = random.randint(0, self.n_users - 1)

        pos_item = np.random.choice(list(self.train_data[user]))
        data_with_negs = np.ones((self.negative_sample_ratio, 3), dtype=np.int64)
        data_with_negs[:, 0] = user
        data_with_negs[:, 1] = pos_item
        data_with_negs[:, 2] = get_negative_items(self, user, self.negative_sample_ratio)
        return data_with_negs

    def output_dataset(self, path):
        if not os.path.exists(path): os.mkdir(path)
        output_inters(os.path.join(path, 'train.txt'), self.train_data)
        output_inters(os.path.join(path, 'val.txt'), self.val_data)
        output_feats(os.path.join(path, 'feats.txt'), self.feats)


class ProcessedDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(ProcessedDataset, self).__init__(dataset_config)
        self.train_data = self.read_data(os.path.join(dataset_config['path'], 'train.txt'))
        self.val_data = self.read_data(os.path.join(dataset_config['path'], 'val.txt'))
        assert len(self.train_data) == len(self.val_data)
        self.n_users = len(self.train_data)

        self.train_array = []
        for user in range(self.n_users):
            self.train_array.extend([[user, item] for item in self.train_data[user]])

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            items = line.split(' ')[1:]
            items = {int(item) for item in items}
            if items:
                self.n_items = max(self.n_items, max(items) + 1)
            data.append(items)
        return data


class AmazonDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(AmazonDataset, self).__init__(dataset_config)

        feats = dict()
        input_file_path = os.path.join(dataset_config['path'], 'meta_Books.jsonl')
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                record = json.loads(line)
                feat = dict()
                feat['title'] = record['title']
                feat['author'] = '' if record.get('author', None) is None else record['author'].get('name', '')
                feat['average_rating'] = record['average_rating']
                feat['rating_number'] = record['rating_number']
                feat['price'] = '' if record['price'] is None else '$' + str(record['price'])
                feat['categories'] = '-'.join(record['categories'][1:])
                pattern = r'^\d+ pages$'
                feat['pages'] = next((v for v in record['details'].values() if type(v) == str and re.match(pattern, v)), '')
                feats[record['parent_asin']] = str(feat)[1:-1]
                line = f.readline().strip()

        input_file_path = os.path.join(dataset_config['path'], 'Books.jsonl')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                record = json.loads(line)
                if record['rating'] > 3 and record['verified_purchase'] == True:
                    update_ui_sets(record['user_id'], record['parent_asin'], user_inter_sets, item_inter_sets)
                line = f.readline().strip()

        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)
        popularity = [None for _ in range(self.n_items)]
        for item, inters in list(item_inter_sets.items()):
            popularity[item_map[item]] = len(inters)

        user_inter_lists = [[] for _ in range(self.n_users)]
        with open(input_file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                record = json.loads(line)
                if record['rating'] > 3 and record['verified_purchase'] == True:
                    update_user_inter_lists(record['user_id'], record['parent_asin'], record['timestamp'],
                                            user_map, item_map, user_inter_lists)
                line = f.readline().strip()
        self.generate_inters(user_inter_lists)
        self.generate_feats(feats, item_map, popularity)


class MINDDataset(BasicDataset):
    def __init__(self, dataset_config, user_sample_ratio=0.1):
        super(MINDDataset, self).__init__(dataset_config)

        feats = dict()
        user_inter_sets, item_inter_sets = dict(), dict()
        news_path = os.path.join(dataset_config['path'], 'news.tsv')
        behaviors_path = os.path.join(dataset_config['path'], 'behaviors.tsv')

        with open(news_path, 'r') as f:
            line = f.readline().strip()
            while line:
                splits = line.split('\t')
                news_id = splits[0]
                category = splits[1]
                subcategory = splits[2]
                title = splits[3]
                abstract = splits[4]
                feat = {
                    'title': title,
                    'category': category,
                    'subcategory': subcategory,
                    'abstract': abstract
                }
                feats[news_id] = str(feat)[1:-1]
                line = f.readline().strip()

        with open(behaviors_path, 'r') as f:
            line = f.readline().strip()
            while line:
                imp_id, user_id, time, history, impressions = line.split('\t')
                if not str_prob(user_id, user_sample_ratio):
                    line = f.readline().strip()
                    continue
                click_items = history.strip().split() + [x.split('-')[0] for x in impressions.strip().split() if x.endswith('-1')]
                for item in click_items:
                    update_ui_sets(user_id, item, user_inter_sets, item_inter_sets)
                line = f.readline().strip()

        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)
        popularity = [None for _ in range(self.n_items)]
        for item, inters in list(item_inter_sets.items()):
            popularity[item_map[item]] = len(inters)

        user_inter_lists = [[] for _ in range(self.n_users)]
        with open(behaviors_path, 'r') as f:
            line = f.readline().strip()
            while line:
                imp_id, user_id, time, history, impressions = line.split('\t')
                if not str_prob(user_id, user_sample_ratio):
                    line = f.readline().strip()
                    continue
                ts = self.time_to_timestamp(time)
                click_items = history.strip().split() + [x.split('-')[0] for x in impressions.strip().split() if x.endswith('-1')]
                for item in click_items:
                    update_user_inter_lists(user_id, item, ts, user_map, item_map, user_inter_lists)
                line = f.readline().strip()

        self.generate_inters(user_inter_lists)
        self.generate_feats(feats, item_map, popularity)

    def time_to_timestamp(self, tstr):
        # tstr: "11/15/2019 10:22:32 AM" -> timestamp
        dt = datetime.strptime(tstr, "%m/%d/%Y %I:%M:%S %p")
        return int(dt.timestamp())
