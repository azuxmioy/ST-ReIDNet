from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np
import re
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from ..utils.serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret = []
    query = {}
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        if relabel:
            if index not in query.keys():
                query[index] = []
        else:
            if pid not in query.keys():
                query[pid] = []
        for i, fname in enumerate(pid_images):
            name = osp.splitext(fname)[0]
            name = name.rsplit('/', 1)[-1]
            x, y = map(int, name.split('_'))
            assert pid == x
            if relabel:
                ret.append((fname, index, y))
                query[index].append(fname)
            else:
                ret.append((fname, pid, y))
                query[pid].append(fname)

    return ret, query

class DanceReid(Dataset):

    def __init__(self, root, split_id=0, num_val=10):
        super(DanceReid, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "Please follow README.md to prepare DanceReid dataset.")

        self.load(num_val)

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        trainval_pids = sorted(np.asarray(self.split['trainval']))
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))

        identities = self.meta['identities']
        self.train, self.train_query = _pluck(identities, train_pids, relabel=True)
        self.val, self.val_query = _pluck(identities, val_pids, relabel=True)
        self.trainval, self.trainval_query = _pluck(identities, trainval_pids, relabel=True)
        self.query, self.query_query = _pluck(identities, self.split['query'])
        self.gallery, self.gallery_query = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))
    

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json')) and \
               osp.isfile(osp.join(self.root, 'video.json')) and \
               osp.isdir(osp.join(self.root, 'poses'))