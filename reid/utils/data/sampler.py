from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
import random
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

def _choose_from(start, end, excluded_range=None, size=1, replace=False):
    num = end - start + 1
    if excluded_range is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluded_range
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds

class RandomPairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1):
        super(RandomPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        # Sort by pid
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # Get the range of indices for each pid
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample
            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluded_range=(i, i))[0]
            yield anchor_index, self.index_map[pos_index]
            # negative samples
            neg_indices = _choose_from(0, self.num_samples - 1,
                                       excluded_range=(start, end),
                                       size=self.neg_pos_ratio)
            for neg_index in neg_indices:
                yield anchor_index, self.index_map[neg_index]

    def __len__(self):
        return self.num_samples * (1 + self.neg_pos_ratio)

class RandomTripletSampler(Sampler):
    def __init__(self, data_source, video_dict=None, skip_frames=10, inter_rate= 0.9, inst_sample = False):
        super(RandomTripletSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.skip_frames = skip_frames
        self.video = video_dict
        self.inst_sample = inst_sample

        if self.video is not None:
            self.rand_list = [True]* int(100*inter_rate) + [False]* int(100*(1-inter_rate))
        else:
            self.rand_list = [False]* 100


        # Sort by pid
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # Get the range of indices for each pid
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

        self.neighbor_map = defaultdict(list)

        if self.video is not None:
            for videoname, pid_list in self.video.items():
                for pid in pid_list:
                    cplist = pid_list.copy()
                    cplist.remove(pid)
                    self.neighbor_map[pid].extend(cplist)


    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample
            # we reject samples which is too close to 
            start, end = self.index_range[pid]

            if self.inst_sample: # Perform instance sampling
                pos_index = _choose_from(start, end, excluded_range=(i, i), size=2, replace=True)
                yield anchor_index, self.index_map[pos_index[0]], self.index_map[pos_index[1]]

            else:   # Perform ST-Sampling
                if random.choice(self.rand_list): # choose STI sampling
                    if (end - start) < (2 * self.skip_frames):
                        skip = (end - start) // 5
                    else:
                        skip = self.skip_frames

                    #pos_index = _choose_from(max (start, i-2*skip) , min (end, i+2*skip),
                        #excluded_range=( max( i-skip, start),  min( i+skip, end) ) )[0]
                    pos_index = _choose_from(start, end, excluded_range=(max (i-skip,start),
                                min(i+skip,end) ) )[0]

                    neg_pid = random.choice(self.neighbor_map[pid])
                    start, end = self.index_range[neg_pid]
                    neg_index = _choose_from(start, end)[0]
                else: # any sample
                    pos_index = _choose_from(start , end, excluded_range=( i, i ) )[0]
                    neg_index = _choose_from(0, self.num_samples - 1,
                                        excluded_range=(start, end))[0] 

                yield anchor_index, self.index_map[pos_index], self.index_map[neg_index]


    def __len__(self):
        return self.num_samples
