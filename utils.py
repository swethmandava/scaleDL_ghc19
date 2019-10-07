# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from functools import reduce
import torch
import torch.nn as nn
import time
from argparse import ArgumentParser
import math
import apex.amp as amp

class DefaultArgs:
    data_folder = "/data/cache/ml-20m"
    epochs = 10
    valid_batch_size = 2**20
    factors = 64
    layers = [256, 256, 128, 64]
    negative_samples = 4
    topk = 10
    threshold = 1.0
    valid_negative = 100
    dropout = 0.1
    seed = 0
    loss_scale = 8192
    local_rank = 0

def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


def generate_neg(users, true_mat, item_range, num_neg, sort=False):
    # assuming 1-d tensor input

    # for each user in 'users', generate 'num_neg' negative samples in [0, item_range)
    # also make sure negative sample is not in true sample set with mask
    # true_mat store a mask matrix where true_mat(user, item) = 0 for true sample
    # return (neg_user, neg_item)

    # list to append iterations of result
    neg_u = []
    neg_i = []

    neg_users = users.repeat(num_neg)
    while len(neg_users) > 0: # generate then filter loop
        neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, item_range)
        neg_mask = true_mat[neg_users, neg_items]
        neg_u.append(neg_users.masked_select(neg_mask))
        neg_i.append(neg_items.masked_select(neg_mask))

        neg_users = neg_users.masked_select(1-neg_mask)

    neg_users = torch.cat(neg_u)
    neg_items = torch.cat(neg_i)
    if sort == False:
        return neg_users, neg_items

    sorted_users, sort_indices = torch.sort(neg_users)
    return sorted_users, neg_items[sort_indices]


def process_data():

    # load not converted data, just seperate one for test
    train_ratings = torch.load(DefaultArgs.data_folder+'/train_ratings.pt', map_location=torch.device('cuda:{}'.format(DefaultArgs.local_rank)))
    test_ratings = torch.load(DefaultArgs.data_folder+'/test_ratings.pt', map_location=torch.device('cuda:{}'.format(DefaultArgs.local_rank)))

    # get input data
    # get dims
    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item()+1
    nb_items = nb_maxs[1].item()+1
    train_users = train_ratings[:,0]
    train_items = train_ratings[:,1]
    del nb_maxs, train_ratings

    # produce things not change between epoch
    # mask for filtering duplicates with real sample
    # note: test data is removed before create mask, same as reference
    mat = torch.cuda.ByteTensor(nb_users, nb_items).fill_(1)
    mat[train_users, train_items] = 0
    # create label
    train_label = torch.ones_like(train_users, dtype=torch.float32)
    neg_label = torch.zeros_like(train_label, dtype=torch.float32)
    neg_label = neg_label.repeat(DefaultArgs.negative_samples)
    train_label = torch.cat((train_label,neg_label))
    del neg_label

    # produce validation negative sample on GPU
    all_test_users = test_ratings.shape[0]

    test_users = test_ratings[:,0]
    test_pos = test_ratings[:,1].reshape(-1,1)
    test_negs = generate_neg(test_users, mat, nb_items, DefaultArgs.valid_negative, True)[1]

    # create items with real sample at last position
    test_users = test_users.reshape(-1,1).repeat(1,1+DefaultArgs.valid_negative)
    test_items = torch.cat((test_negs.reshape(-1, DefaultArgs.valid_negative), test_pos), dim=1)
    del test_ratings, test_negs

    # generate dup mask and real indice for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    # this is a version works on integer
    sorted_items, indices = torch.sort(test_items) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0]) #[1.75,1.25,1.0,2.5]
    indices_order = torch.sort(sum_item_indices)[1] #[2,1,0,3]
    stable_indices = torch.gather(indices, 1, indices_order) #[0,1,3,2]
    # produce -1 mask
    dup_mask = (sorted_items[:,0:-1] == sorted_items[:,1:])
    dup_mask = torch.cat((torch.zeros_like(test_pos, dtype=torch.uint8), dup_mask),dim=1)
    dup_mask = torch.gather(dup_mask,1,stable_indices.sort()[1])
    # produce real sample indices to later check in topk
    sorted_items, indices = (test_items != test_pos).sort()
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0])
    indices_order = torch.sort(sum_item_indices)[1]
    stable_indices = torch.gather(indices, 1, indices_order)
    real_indices = stable_indices[:,0]
    del sorted_items, indices, sum_item_indices, indices_order, stable_indices, test_pos

    return train_label, train_users, train_items, test_users, test_items, dup_mask, real_indices, all_test_users, nb_users, nb_items, mat

def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user):
    log_2 = math.log(2)

    model.eval()
    with torch.no_grad():
        p = []
        for u, n in zip(x, y):
            p.append(model(u, n, sigmoid=True).detach())

        del x
        del y
        temp = torch.cat(p).view(-1, samples_per_user)
        del p
        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp, K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(predicetion) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1, 1))
        hits = ifzero.sum()
        ndcg = (log_2 / (torch.nonzero(ifzero)[:, 1].view(-1).to(torch.float) + 2).log_()).sum()

    hits = hits.item()
    ndcg = ndcg.item()

    model.train()
    return hits / num_user, ndcg / num_user


def train(model, optimizer, processed_data, batch_size, learning_rate, mode="train", warmup_fn=None, scale_loss_fn=None):

    train_label, train_users, train_items, test_users, test_items, \
    dup_mask, real_indices, all_test_users, \
    nb_users, nb_items, mat = processed_data

    train_users_per_worker = len(train_label)
    train_users_begin = 0

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU
    criterion = criterion.cuda()

    print("\nOpt Learning rate %0.4f" %learning_rate)
    print("Input Batch Size %d" %batch_size)
    success = False
    max_hr = 0
    local_batch = batch_size
    train_throughputs = []
    eval_throughputs = []
    traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch, 1), torch.rand(local_batch, 1)))
    main_start_time = time.time()
    for epoch in range(DefaultArgs.epochs):

        print("\nEpoch = %d" % epoch)

        begin = time.time()

        # prepare data for epoch
        neg_users, neg_items = generate_neg(train_users, mat, nb_items, DefaultArgs.negative_samples)
        epoch_users = torch.cat((train_users, neg_users))
        epoch_items = torch.cat((train_items, neg_items))

        del neg_users, neg_items

        # shuffle prepared data and split into batches
        epoch_indices = torch.randperm(train_users_per_worker, device='cuda:{}'.format(DefaultArgs.local_rank))
        epoch_indices += train_users_begin

        epoch_users = epoch_users[epoch_indices]
        epoch_items = epoch_items[epoch_indices]
        epoch_label = train_label[epoch_indices]

        epoch_users_list = epoch_users.split(local_batch)
        epoch_items_list = epoch_items.split(local_batch)
        epoch_label_list = epoch_label.split(local_batch)

        # only print progress bar on rank 0
        num_batches = len(epoch_users_list)

        for batch_idx in range(num_batches):
            if warmup_fn is not None:
                iter_i = epoch * num_batches + batch_idx
                warmup_fn(optimizer, iter_i, num_batches)
            user = epoch_users_list[batch_idx]
            item = epoch_items_list[batch_idx]
            label = epoch_label_list[batch_idx].view(-1, 1)
            optimizer.zero_grad()
            outputs = model(user, item)
            loss = traced_criterion(outputs, label).float()
            loss = torch.mean(loss.view(-1), 0)

            if scale_loss_fn is None:
                loss.backward()
            else:
                scale_loss_fn(optimizer, loss)

            optimizer.step()
            for p in model.parameters():
                p.grad = None

        del epoch_users, epoch_items, epoch_label, epoch_users_list, epoch_items_list, epoch_label_list, user, item, label
        train_time = time.time() - begin
        begin = time.time()

        epoch_samples = len(train_users) * (DefaultArgs.negative_samples + 1)
        train_throughput = epoch_samples / train_time
        train_throughputs.append(train_throughput)
        print("Train Throughput = %0.4f" %train_throughput)

        hr, ndcg = val_epoch(model, test_users.view(-1).split(DefaultArgs.valid_batch_size), test_items.view(-1).split(DefaultArgs.valid_batch_size), dup_mask, real_indices, DefaultArgs.topk,
                             samples_per_user=test_items.size(1),
                             num_user=all_test_users)
        val_time = time.time() - begin

        if mode == "perf":
            break

        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=DefaultArgs.topk, hit_rate=hr,
                      ndcg=ndcg, train_time=train_time,
                      val_time=val_time))

        if hr > max_hr and DefaultArgs.local_rank == 0:
            max_hr = hr
            print("New best hr!")

        if DefaultArgs.threshold is not None:
            if hr >= DefaultArgs.threshold:
                print("Hit threshold of {}".format(DefaultArgs.threshold))
                success = True
                break
    if mode == "train":
        print("\nBest train throughput %0.4f" %max(train_throughputs))
        print("Best accuracy %0.4f" %max_hr)
        print("Time to Target %0.4f" %(time.time() - main_start_time))

    return max_hr