import argparse
import os, sys
import os.path as osp

import model.utils.util as util
from reid import models
from reid import datasets

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--stage', type=int, default=0, help='training stage [0|1|2](0 is end-to-end, 1 is train only resnet, and 2 is fintuning)')
        self.parser.add_argument('-d', '--dataset', type=str, default='DanceReID', choices=datasets.names())
        # paths

        self.parser.add_argument('--dataroot', type=str, default='./datasets/', help='root path to datasets (should have subfolders market1501, dukemtmc, DanceReID, etc)')
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='root path to save models')

        self.parser.add_argument('--name', type=str, default='my_session', help='directory to save models')
        self.parser.add_argument('--netE-pretrain', type=str, default='', help='pretrained model path for net_E in stage 2')
        self.parser.add_argument('--netG-pretrain', type=str, default='', help='pretrained model path for net_G in stage 2')
        self.parser.add_argument('--netDp-pretrain', type=str, default='', help='pretrained model path for net_Dp in stage 2')
        self.parser.add_argument('--netDi-pretrain', type=str, default='', help='pretrained model path for net_Di in stage 2')

        # model structures
        self.parser.add_argument('--arch', type=str, default='resnet50', help='basemodeltype')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--drop', type=float, default=0.2, help='dropout for the netG')
        self.parser.add_argument('--connect-layers', type=int, default=0, help='skip connections num for netG')
        self.parser.add_argument('--fuse-mode', type=str, default='cat', help='method to fuse reid feature and pose feature [cat|add]')
        self.parser.add_argument('--pose-feature-size', type=int, default=128, help='length of feature vector for pose')
        self.parser.add_argument('--noise-feature-size', type=int, default=256, help='length of feature vector for noise')
        self.parser.add_argument('--pose-aug', type=str, default='no', help='posemap augmentation [no|erase|gauss]')
        
        # dataloader setting
        self.parser.add_argument('-b', '--batch-size', type=int, default=16, help='input batch size')
        self.parser.add_argument('-j', '--workers', default=4, type=int, help='num threads for loading data')
        self.parser.add_argument('--width', type=int, default=128, help='input image width')
        self.parser.add_argument('--height', type=int, default=256, help='input image height')

        self.parser.add_argument('--skip-frame', type=int, default=20, help='ST-sampling skipping frames')
        self.parser.add_argument('--inter-rate', type=float, default=0.7, help='probability to sample same video (ST-sampling)')

        # optimizer setting
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter-decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--save-step', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--eval-step', type=int, default=5, help='frequency of evaluate checkpoints at the end of epochs')
        
        # visualization setting
        self.parser.add_argument('--display-port', type=int, default=6006, help='visdom port of the web display')
        self.parser.add_argument('--display-id', type=int, default=1, help='window id of the web display, set 0 for non-usage of visdom')
        self.parser.add_argument('--display-winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display-freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--display-single-pane-ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update-html-freq', type=int, default=100, help='frequency of saving training results to html')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints]/name/web/')
        self.parser.add_argument('--print-freq', type=int, default=10, help='frequency of showing training results on console')
        
        # training
        self.parser.add_argument('--lambda-dp', type=float, default=1.0, help='weight of Dp')
        self.parser.add_argument('--lambda-d', type=float, default=1.0, help='weight of D')
        self.parser.add_argument('--lambda-recon', type=float, default=1.0, help='loss weight of reconstruction')
        self.parser.add_argument('--lambda-tri', type=float, default=1.0, help='loss weight of triplet loss')
        self.parser.add_argument('--lambda-class', type=float, default=1.0, help='loss weight of classification')
        self.parser.add_argument('--smooth-label', action='store_true', help='smooth label or not for GANloss')
        self.parser.add_argument('--margin', type=float, default=1.0, help='margin for triplet loss')
        self.parser.add_argument('--soft-margin', action='store_true', help='use soft margin for triplet loss')
        self.parser.add_argument('--batch-hard', action='store_true', help='batch hard mining for tri-loss')
        self.parser.add_argument('--emb-type', type=str, default='Single', help='ReID embedding model type [Single|Siamese]')

        # apply training tricks
        self.parser.add_argument('--mask', action='store_true', help='use pose mask for human reconstruction or not')
        self.parser.add_argument('--eraser', action='store_true', help='use random eraser during training')
        self.parser.add_argument('--emb-smooth', action='store_true', help='use label smoothing for computing classification loss')
        self.parser.add_argument('--last-stride', type=int, default=2, help='resnet50 last stride')

        self.parser.add_argument('--seed', type=int, default=1)


        self.opt = self.parser.parse_args()
        self.show_opt()

    def parse(self):
        return self.opt

    def show_opt(self):
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
