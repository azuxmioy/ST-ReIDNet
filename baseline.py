from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import os, sys
from bisect import bisect_right
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, read_json

from reid.utils.data.sampler import RandomPairSampler, RandomTripletSampler
from reid.models.embedding import EltwiseSubEmbed, BNClassifierEmbed
from reid.models.multi_branch import SiameseNet, SingleNet
from reid.evaluators import CascadeEvaluator
from reid.trainers import SiameseTrainer, TripletTrainer

from model.losses import TripletLoss, CrossEntropyLabelSmooth
    
def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, np_ratio, model, instance_mode, eraser):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train

    if eraser:
        train_transformer = T.Compose([
            T.RandomSizedRectCrop(height, width),
            T.RandomSizedEarser(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
    else:
        train_transformer = T.Compose([
            T.RandomSizedRectCrop(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    if (model == 'Single'):
        video_dict = None
        if osp.isfile(osp.join(root, 'video.json')):
            video_dict = read_json(osp.join(root, 'video.json'))
        sampler = RandomTripletSampler(train_set, video_dict=None, skip_frames=10, inter_rate= 0.9, inst_sample = instance_mode)
    elif (model== 'Siamese'):  
        sampler = RandomPairSampler(train_set, neg_pos_ratio=np_ratio)
    else:
        raise ValueError('unrecognized mode')


    train_loader = DataLoader(
        Preprocessor(train_set, name, root=dataset.images_dir,
                     transform=train_transformer),
                     sampler=sampler, batch_size=batch_size,
                     num_workers=workers, pin_memory=False)

    val_loader = DataLoader(
        Preprocessor(dataset.val, name,  root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)), name,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, args.np_ratio,
                args.emb_type, args.inst_mode, args.eraser)

    if args.combine_trainval:
        emb_size = dataset.num_trainval_ids
    else:
        emb_size = dataset.num_train_ids

    # Create model
    if (args.emb_type == 'Single'):
        model = SingleNet(args.arch, emb_size, pretraind=True, use_bn=args.use_bn, test_bn=args.test_bn, last_stride=args.last_stride)
    elif  (args.emb_type == 'Siamese'):  
        model = SiameseNet(args.arch, emb_size, pretraind=True, use_bn=args.use_bn, test_bn=args.test_bn, last_stride=args.last_stride)
    else:
        raise ValueError('unrecognized model')
    model = nn.DataParallel(model).cuda()

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)

    # Evaluator

    evaluator = CascadeEvaluator(torch.nn.DataParallel(model).cuda(), emb_size=emb_size)

    # Load from checkpoint
    best_mAP = 0
    if args.resume:
        print("Test the loaded model:")
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, dataset=args.dataset)
        best_mAP = mAP
    if args.evaluate:
        return

    # Criterion
    if args.soft_margin:
        tri_criterion = TripletLoss(margin='soft').cuda()
    else:
        tri_criterion = TripletLoss(margin=args.margin).cuda()

    if (args.emb_type == 'Single'):
        if args.label_smoothing:
            cla_criterion = CrossEntropyLabelSmooth(emb_size, epsilon=0.1).cuda()
        else:
            cla_criterion = torch.nn.CrossEntropyLoss().cuda()
    elif  (args.emb_type == 'Siamese'):  
        cla_criterion = torch.nn.CrossEntropyLoss().cuda()

    # Optimizer
    param_groups = [
        {'params': model.module.base_model.parameters(), 'lr_mult': 0.1},
        {'params': model.module.classifier.parameters(), 'lr_mult': 1.0}]

    if (args.opt_name == 'SGD'):
        optimizer = getattr(torch.optim, args.opt_name) (param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = getattr(torch.optim, args.opt_name) (param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Trainer

    if (args.emb_type == 'Single'):
        trainer = TripletTrainer(model, tri_criterion, cla_criterion, args.lambda_tri, args.lambda_cla)
    elif  (args.emb_type == 'Siamese'):  
        trainer = SiameseTrainer(model, tri_criterion, cla_criterion, args.lambda_tri, args.lambda_cla)

    #TODO:Warmup lr
    # Schedule learning rate
    def adjust_lr(epoch):

        lr = args.lr * (0.1 ** (epoch // args.step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)


    # Start training
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, base_lr=args.lr)

        if epoch % args.eval_step==0:
            #mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, top1=False, dataset=args.dataset)
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, top1=False, dataset=args.dataset)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict()
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DanceReID baseline")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='DanceReID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--np-ratio', type=int, default=3)

    parser.add_argument('--inst-mode', action='store_true',
                        help="perform instance sampling")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--emb-type', type=str, default='Single')

    #losses
    parser.add_argument('--soft-margin', action='store_true', help='use smooth margin for triplet loss')
    parser.add_argument('--margin', type=int, default=4)
    parser.add_argument('--lambda-tri', type=float, default=1.0, help='weight of Triplet loss')
    parser.add_argument('--lambda-cla', type=float, default=1.0, help='weight of Classificatin loss')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--opt-name', type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    
    # train/val configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size', type=int, default=40)
    parser.add_argument('--eval-step', type=int, default=20, help="evaluation step")
    parser.add_argument('--seed', type=int, default=1)


    # Training/trick
    parser.add_argument('--eraser', action='store_true', help='use random eraser for data augmentation')
    parser.add_argument('--label-smoothing', action='store_true', help='use smooth label for classify')
    parser.add_argument('--last-stride', type=int, default=1, help='ResNet last stride 1 or 2')
    parser.add_argument('--use-bn', action='store_true', help='bn before classifier')
    parser.add_argument('--test-bn', action='store_true', help='use bn feature for eval')

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'checkpoints'))
    main(parser.parse_args())
