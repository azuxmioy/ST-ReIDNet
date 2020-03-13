from __future__ import absolute_import
import os.path as osp
import random
import numpy as np
import torch
from scipy import ndimage
from PIL import Image
import cv2

from reid.utils.data import transforms


DICT_POSE_17 = dict()
DICT_POSE_17['head'] = [
    (0, 1), (0, 2), (1, 3), (2, 4)]
DICT_POSE_17['hand'] = [
    (5, 7), (7, 9), (6, 8), (8, 10)]
DICT_POSE_17['legs'] = [
    (11, 13), (12, 14), (13, 15), (14, 16)]  
DICT_POSE_17['body'] = [
    (5, 6), (5, 11), (6, 12), (11, 12), (6, 11), (5, 12)] # Body
DICT_POSE_17['neck'] = [
    (0, 6), (0, 5)]


DICT_POSE_18 = dict()
DICT_POSE_18['head'] = [
    (16, 14), (14, 0), (0, 15), (15, 17)]
DICT_POSE_18['hand'] = [
    (4, 3), (3, 2), (5, 6), (6, 7)]
DICT_POSE_18['legs'] = [
    (8, 9), (9, 10), (11, 12), (12, 13)]  
DICT_POSE_18['body'] = [
    (2, 1), (1, 5), (1, 8), (1, 11), (2, 8),
    (11, 8), (5, 11), (11, 2), (5, 18)] # Body
DICT_POSE_18['neck'] = [
    (0, 1)]


class Preprocessor(object):
    def __init__(self, dataset, name, root=None, with_pose=False, pose_root=None, is_test=False, test_root = None,
                 pid_imgs=None, height=256, width=128, pose_aug='no', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.with_pose = with_pose
        self.pose_root = pose_root
        self.is_test = is_test
        self.test_root = test_root
        self.pid_imgs = pid_imgs
        self.height = height
        self.width = width
        self.pose_aug = pose_aug
        self.name = name

        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform==None:
            self.transform = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.RandomSizedEarser(),
                                 #transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])
        else:
            self.transform = transform
        self.transform_p = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if self.is_test:
                return [self._get_single_item_test(index) for index in indices]
            elif not self.with_pose:
                return [self._get_single_item(index) for index in indices]
            else:
                return [self._get_single_item_with_pose(index) for index in indices]
        if self.is_test:
            return self._get_single_item_test(indices)
        elif not self.with_pose:
            return self._get_single_item(indices)
        else:
            return self._get_single_item_with_pose(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid

    def _get_single_item_test(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.test_root is not None:
            fpath = osp.join(self.test_root, fname)
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid


    def _get_single_item_with_pose(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        ppath = osp.splitext(fname)[0]+'.txt'
        if self.pose_root is not None:
            ppath = osp.join(self.pose_root, ppath)

        '''
        img = self.transform(img)

        pid_query = list(self.pid_imgs[pid])
        if fname in pid_query and len(pid_query)>1:
            pid_query.remove(fname)
        pname = osp.splitext(random.choice(pid_query))[0]

        ppath = pname+'.txt'
        if self.pose_root is not None:
            ppath = osp.join(self.pose_root, ppath)
        gtpath = pname+'.jpg'
        if self.root is not None:
            gtpath = osp.join(self.root, gtpath)

        gt_img = Image.open(gtpath).convert('RGB')
        '''
        landmark = self._load_landmark(ppath, self.height/img.size[1], self.width/img.size[0])

        maps = self._generate_pose_map(landmark)

        mask = self._generate_human_mask(landmark)

        flip_flag = random.choice([True, False])
        if flip_flag:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            maps = np.flip(maps, 2)
            mask = np.flip(mask, 1)

        maps = torch.from_numpy(maps.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        gt = self.transform_p(img)
        erase = self.transform(img)

        return {'origin': gt,
                'input': erase,
                'posemap': maps,
                'mask':mask,
                'pid': torch.LongTensor([pid])}

    def _load_landmark(self, path, scale_h, scale_w):
        landmark = []
        with open(path,'r') as f:
            landmark_file = f.readlines()
        for line in landmark_file:
            line1 = line.strip()
            if (self.name == 'DanceReID'):
                h0 = int(float(line1.split(' ')[1]) * scale_h)
                w0 = int(float(line1.split(' ')[0]) * scale_w)
            else:
                h0 = int(float(line1.split(' ')[0]) * scale_h)
                w0 = int(float(line1.split(' ')[1]) * scale_w)               
            if h0<0 or h0>=self.height: h0=-1
            if w0<0 or w0>=self.width: w0=-1
            landmark.append(torch.Tensor([[h0,w0]]))
        landmark = torch.cat(landmark).long()
        return landmark

    def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug=='erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug=='gauss':
            gauss_sigma = random.randint(gauss_sigma-1,gauss_sigma+1)
        elif self.pose_aug!='no':
            assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
        for i in range(landmark.size(0)):
            map = np.zeros([self.height,self.width])
            if landmark[i,0]!=-1 and landmark[i,1]!=-1 and i!=randnum:
                map[landmark[i,0],landmark[i,1]]=1
                map = ndimage.filters.gaussian_filter(map,sigma = gauss_sigma)
                map = map/map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        return maps

    def _generate_human_mask(self, landmark):

        dict_pose_line = dict()
        pair_dict = dict()
        if self.name == 'DanceReID':
            pair_dict = DICT_POSE_17
        else:
            pair_dict = DICT_POSE_18

        mask = np.zeros((self.height,self.width,3), dtype=np.uint8)
        binary = np.zeros((self.height,self.width))
        for i in range(landmark.size(0)):

            y, x = landmark[i, 0], landmark[i, 1]
            if x!=-1 and y!=-1:
                dict_pose_line[i] = (x, y)
                radius = 10
                if (i==0): radius = 20
                cv2.circle(mask, (x, y), radius=radius, color=(255,255,255), thickness=-1)

        # Head mask:
        for start_p, end_p in pair_dict['head']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=20)

        # Hand mask:
        for start_p, end_p in pair_dict['hand']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=20)
        # Hand mask:
        for start_p, end_p in pair_dict['legs']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=20)
        # Hand mask:
        for start_p, end_p in pair_dict['body']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=30)
        # Hand mask:
        for start_p, end_p in pair_dict['neck']:
            if start_p in dict_pose_line and end_p in dict_pose_line:
                start_p = dict_pose_line[start_p]
                end_p = dict_pose_line[end_p]
                cv2.line(mask, start_p, end_p, color=(255,255,255), thickness=30)



        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary[mask>128] = 1
        return binary.astype(np.float)
