from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter

class BaseTrainer(object):
    def __init__(self, model, criterion_tri, criterion_cla, lambda_tri=1.0, lambda_cla=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion_tri = criterion_tri
        self.criterion_cla = criterion_cla
        self.lambda_tri = lambda_tri
        self.lambda_cla = lambda_cla

    def train(self, epoch, data_loader, optimizer, base_lr=0.1, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets, ids = self._parse_data(inputs)
            #c_loss, t_loss, loss, prec1, outputs= self._forward(inputs, targets)
            c_loss, t_loss, loss, prec1, outputs = self._forward(inputs, targets, ids)

            losses.update(loss.data, ids.size(0))
            precisions.update(prec1, ids.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            step = epoch * len(data_loader) + i + 1

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_c {:.3f} Loss_t{:.3f} \t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              c_loss, t_loss,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, ids):
        raise NotImplementedError

class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = torch.cat([Variable(imgs1), Variable(imgs2)])
        targets = Variable((pids1 == pids2).long().cuda())
        ids = torch.cat([pids1, pids2]).long().cuda()
        return inputs, targets, ids

    def _forward(self, inputs, targets, ids):
        A1, A2, outputs = self.model(inputs)
        loss_c = self.criterion_cla(outputs, torch.squeeze(targets))
        prec1, = accuracy(outputs.data, targets.data)

        feat = torch.cat([A1, A2])
        loss_t, _ , _ = self.criterion_tri(feat, torch.squeeze(ids))

        loss = loss_c * self.lambda_cla + loss_t * self.lambda_tri

        return loss_c, loss_t, loss, prec1[0], outputs

#    with pose {'origin': img, 'target': gt_img,'posemap': maps,'pid': torch.LongTensor([pid])}
#    w/o pose   img, fname, pid, camid

class TripletTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs_ach, _, pid_ach, _), (imgs_pos, _, pids_pos, _), (imgs_neg, _, pid_neg, _) = inputs

        inputs = [Variable(imgs_ach), Variable(imgs_pos), Variable(imgs_neg)]

        targets = None

        ids = Variable(torch.cat([pid_ach, pids_pos, pid_neg]).long().cuda())
        return inputs, targets, ids

    def _forward(self, inputs, targets, ids):

        e_ach, pred_ach = self.model(inputs[0])
        e_pos, pred_pos = self.model(inputs[1])
        e_neg, pred_neg = self.model(inputs[2])

        feat = torch.cat([e_ach, e_pos, e_neg])
        outputs = torch.cat([pred_ach, pred_pos, pred_neg])

        loss_t, _ , _ = self.criterion_tri(feat, torch.squeeze(ids))
        loss_c = self.criterion_cla(outputs, ids)
        loss = loss_c * self.lambda_cla + loss_t * self.lambda_tri

        prec1, = accuracy(outputs.data, ids.data)

        return loss_c, loss_t, loss, prec1[0], outputs
