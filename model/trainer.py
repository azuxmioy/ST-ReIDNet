import os,sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import model.utils.util as util
from reid.utils.serialization import load_checkpoint

from model.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, \
                            remove_module_key, set_bn_fix, get_scheduler, print_network
from model.losses import GANLoss, TripletLoss, CrossEntropyLabelSmooth, MaskedL1loss
from reid.models.embedding import EltwiseSubEmbed, BNClassifierEmbed
from reid.models.multi_branch import SiameseNet, SingleNet

class ST_ReIDNet(object):

    def __init__(self, opt, emb_size):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)
        self.emb_size = emb_size

        if (self.opt.dataset == 'DanceReID'):
            self.pose_size = 17
        else:
            self.pose_size = 18
        
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        print('---------- Networks initialized -------------')
        #print_network(self.net_E)
        #print_network(self.net_G)
        #print_network(self.net_Di)
        #print_network(self.net_Dp)
        #print('-----------------------------------------------')

    def _init_models(self):
        #self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, 0, pose_nc=self.pose_size, dropout=self.opt.drop,
                                         norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode,
                                         connect_layers=self.opt.connect_layers)

        if (self.opt.emb_type == 'Single'):
            self.net_E = SingleNet(self.opt.arch, self.emb_size, pretraind=True, use_bn=True, test_bn=False, last_stride=self.opt.last_stride)
        elif  (self.opt.emb_type == 'Siamese'):  
            self.net_E = SiameseNet(self.opt.arch, self.emb_size, pretraind=True, use_bn=True, test_bn=False, last_stride=self.opt.last_stride)
        else:
            raise ValueError('unrecognized model')

        self.net_Di = SingleNet('resnet18', 1, pretraind=True, use_bn=True, test_bn=False, last_stride=2)

        self.net_Dp = NLayerDiscriminator(3+self.pose_size, norm_layer=self.norm_layer)

        if self.opt.stage==0: # This is for training end-to-end
            init_weights(self.net_G)
            init_weights(self.net_Dp)
        elif self.opt.stage==1: # This is for training fixing a baseline model
            init_weights(self.net_G)
            init_weights(self.net_Dp)
            checkpoint = load_checkpoint(self.opt.netE_pretrain)
            
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            state_dict = remove_module_key(checkpoint)

            self.net_E.load_state_dict(state_dict)
            #state_dict['classifier.weight'] = state_dict['classifier.weight'][1:2]
            #state_dict['classifier.bias'] = torch.FloatTensor([state_dict['classifier.bias'][1]])
            #self.net_Di.load_state_dict(state_dict)
        elif self.opt.stage==2: # This is for training with a provided model
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
        else:
            raise ValueError('unrecognized mode')

        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_G = torch.nn.DataParallel(self.net_G).cuda()
        self.net_Di = torch.nn.DataParallel(self.net_Di).cuda()
        self.net_Dp = torch.nn.DataParallel(self.net_Dp).cuda()

    def reset_model_status(self):
        if self.opt.stage==0:
            self.net_E.train()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()
            self.net_Di.apply(set_bn_fix)
        elif self.opt.stage==1:
            self.net_G.train()
            self.net_Dp.train()
            self.net_E.eval()
            self.net_Di.train()
            self.net_Di.apply(set_bn_fix)
        elif self.opt.stage==2:
            self.net_E.train()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()
            self.net_E.apply(set_bn_fix)
            self.net_Di.apply(set_bn_fix)

    def _load_state_dict(self, net, path):

        state_dict = remove_module_key(torch.load(path))
        net.load_state_dict(state_dict)

    def _init_losses(self):
        if self.opt.smooth_label:
            self.criterionGAN_D = GANLoss(smooth=True).cuda()
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.criterionGAN_D = GANLoss(smooth=False).cuda()
            self.rand_list = [False]
        self.criterionGAN_G = GANLoss(smooth=False).cuda()

        if self.opt.soft_margin:
            self.tri_criterion = TripletLoss(margin='soft', batch_hard=True, distractor=True).cuda()
        else:
            self.tri_criterion = TripletLoss(margin=self.opt.margin, batch_hard=True, distractor=True).cuda

        if (self.opt.emb_type == 'Single'):
            if (self.opt.emb_smooth):
                self.class_criterion = CrossEntropyLabelSmooth(self.emb_size, epsilon=0.1).cuda()
            else:
                self.class_criterion = torch.nn.CrossEntropyLoss().cuda()
        elif  (self.opt.emb_type == 'Siamese'): 
            self.class_criterion = torch.nn.CrossEntropyLoss().cuda()

        if self.opt.mask:
            self.reco_criterion = MaskedL1loss(use_mask=True).cuda()
        else:
            self.reco_criterion = MaskedL1loss(use_mask=False).cuda()



    def _init_optimizers(self):
        if self.opt.stage==0:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.classifier.parameters(), 'lr_mult': 1.0}]
            self.optimizer_E = torch.optim.SGD(param_groups, lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)
            #self.optimizer_E = torch.optim.Adam(param_groups, lr=self.opt.lr, betas=(0.9, 0.999), weight_decay=5e-4)
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=1e-5, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=4e-5, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=4e-5, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage==1:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.classifier.parameters(), 'lr_mult': 0.1}]

            self.optimizer_E = torch.optim.SGD(param_groups, lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)

            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage==2:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.01},
                            {'params': self.net_E.module.classifier.parameters(), 'lr_mult': 0.1}]

            self.optimizer_E = torch.optim.SGD(param_groups, lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)

            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=1e-6, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=1e-5, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=1e-5, momentum=0.9, weight_decay=1e-4)


        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optimizer_E)
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))

    def set_input(self, input):
        input_ach, input_pos, input_neg = input
        ids = torch.cat([input_ach['pid'], input_pos['pid'], input_neg['pid']]).long()

        ones = torch.ones_like(input_ach['pid']).fill_(1.0)
        zeros = torch.ones_like(input_ach['pid']).fill_(0.0)

        labels = torch.cat([ones, zeros]).long()

        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)
        noise = torch.cat((noise, noise, noise))



        self.ach_gt = input_ach['origin'].cuda()
        self.pos_gt = input_pos['origin'].cuda()
        self.neg_gt = input_neg['origin'].cuda()

        self.ach_img = input_ach['input'].cuda()
        self.pos_img = input_pos['input'].cuda()
        self.neg_img = input_neg['input'].cuda()

        self.ach_pose = input_ach['posemap'].cuda()
        self.pos_pose = input_pos['posemap'].cuda()
        self.neg_pose = input_neg['posemap'].cuda()

        self.ach_mask = input_ach['mask'].cuda()
        self.pos_mask = input_pos['mask'].cuda()
        self.neg_mask = input_neg['mask'].cuda()


        self.gt = torch.cat( [self.ach_gt, self.pos_gt, self.neg_gt])
        self.mask = torch.cat( [self.ach_mask, self.pos_mask, self.neg_mask])

        self.posemap = torch.cat( [self.ach_pose, self.pos_pose, self.neg_pose])

        self.two_gt = torch.cat([self.ach_gt, self.pos_gt])
        self.two_mask = torch.cat( [self.ach_mask, self.pos_mask])

        self.pos_posemap = torch.cat([self.ach_pose, self.pos_pose])
        self.swap_posemap = torch.cat([self.pos_pose, self.ach_pose])

        self.ids = ids.cuda()
        self.labels = labels.cuda()
        self.noise = noise.cuda()

    def forward(self):
        z = Variable(self.noise)


        if (self.opt.emb_type == 'Single'):
            if (self.opt.stage == 1):
                self.A_ach = self.net_E(Variable(self.ach_img))
                self.A_pos = self.net_E(Variable(self.pos_img))
                self.A_neg = self.net_E(Variable(self.neg_img))
            else:
                self.A_ach, pred_ach = self.net_E(Variable(self.ach_img))
                self.A_pos, pred_pos = self.net_E(Variable(self.pos_img))
                self.A_neg, pred_neg = self.net_E(Variable(self.neg_img))
                self.id_pred = torch.cat( [pred_ach, pred_pos, pred_neg])

        elif  (self.opt.emb_type == 'Siamese'):
            self.A_ach, self.A_pos, pos_pred = self.net_E(torch.cat([Variable(self.ach_img), Variable(self.pos_img)]))
            _, self.A_neg, neg_pred = self.net_E(torch.cat([Variable(self.ach_img), Variable(self.neg_img)]))
            self.id_pred =  torch.cat( [pos_pred, neg_pred])

        self.fake_ach = self.net_G(Variable(self.ach_pose),
                 self.A_ach.view(self.A_ach.size(0), self.A_ach.size(1), 1, 1), None)
        self.fake_pos = self.net_G(Variable(self.pos_pose),
                 self.A_pos.view(self.A_pos.size(0), self.A_pos.size(1), 1, 1), None)
        self.fake_neg = self.net_G(Variable(self.neg_pose),
                 self.A_neg.view(self.A_neg.size(0), self.A_neg.size(1), 1, 1), None)

        self.swap_ach = self.net_G(Variable(self.ach_pose),
                 self.A_pos.view(self.A_pos.size(0), self.A_pos.size(1), 1, 1), None)
        self.swap_pos = self.net_G(Variable(self.pos_pose),
                 self.A_ach.view(self.A_ach.size(0), self.A_ach.size(1), 1, 1), None)

        self.fake = torch.cat((self.fake_ach, self.fake_pos, self.fake_neg))

        self.swap_fake = torch.cat((self.swap_ach, self.swap_pos))


    def backward_Dp(self):
        real_pose =  torch.cat((Variable(self.posemap), Variable(self.gt)), dim=1)
        fake_pose1 = torch.cat((Variable(self.posemap), self.fake.detach()), dim=1)
        fake_pose2 = torch.cat((Variable(self.pos_posemap), self.swap_fake.detach()), dim=1)
        fake_pose3 = torch.cat((Variable(self.swap_posemap), Variable(self.two_gt)), dim=1)

        
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(torch.cat([fake_pose1, fake_pose2, fake_pose3]))
        #print(pred_real.size())
        #print(pred_fake.size())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        #self.loss_Dp = loss_D.data[0]
        self.loss_Dp = loss_D.item()

    def backward_Di(self):
        _, pred_real = self.net_Di(Variable(self.gt))
        _, pred_fake1 = self.net_Di(self.fake.detach())
        _, pred_fake2 = self.net_Di(self.swap_fake.detach())

        pred_fake = torch.cat ([pred_fake1, pred_fake2])

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        #self.loss_Di = loss_D.data[0]
        self.loss_Di = loss_D.item()


    def backward_G(self):
        
        loss_r1 = self.reco_criterion(self.fake, Variable(self.mask), Variable(self.gt))
        loss_r2 = self.reco_criterion(self.swap_fake, Variable(self.two_mask), Variable(self.two_gt))

        loss_r = 0.7 * loss_r2 + 0.3 * loss_r1

        _, pred_fake_Di = self.net_Di(torch.cat([self.fake, self.swap_fake]))

        fake_pose1 = torch.cat((Variable(self.posemap), self.fake), dim=1)
        fake_pose2 = torch.cat((Variable(self.pos_posemap), self.swap_fake), dim=1)
        fake_pose3 = torch.cat((Variable(self.swap_posemap), Variable(self.two_gt)), dim=1)

        pred_fake_Dp = self.net_Dp(torch.cat([fake_pose1, fake_pose2, fake_pose3]))
        
        loss_G_GAN_Di = self.criterionGAN_G(pred_fake_Di, True)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)


        loss_G = loss_G_GAN_Di * self.opt.lambda_d + \
                 loss_G_GAN_Dp * self.opt.lambda_dp  + \
                 loss_r * self.opt.lambda_recon

        loss_G.backward(retain_graph=True)

        # Compute triplet loss
        feat = torch.cat([self.A_ach, self.A_pos, self.A_neg])
        loss_t, _ , _ = self.tri_criterion(feat, torch.squeeze(self.ids, 1))
        # Classification loss
        if (self.opt.stage==1):
            loss_c = torch.tensor(0.0)
        else:
            if (self.opt.emb_type == 'Single'):
                loss_c = self.class_criterion(self.id_pred, torch.squeeze(self.ids, 1))
            elif (self.opt.emb_type == 'Siamese'):
                loss_c = self.class_criterion(self.id_pred, torch.squeeze(self.labels, 1))


        loss_E = loss_G + \
                loss_t * self.opt.lambda_tri+ \
                loss_c * self.opt.lambda_class
                
        loss_E.backward()

        self.loss_E = loss_E.item()
        self.loss_t = loss_t.item()
        self.loss_c = loss_c.item()

        self.loss_G = loss_G.item()
        self.loss_r = loss_r.item()
        self.loss_G_GAN_Di = loss_G_GAN_Di.item()
        self.loss_G_GAN_Dp = loss_G_GAN_Dp.item()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_Di.zero_grad()
        self.backward_Di()
        self.optimizer_Di.step()

        self.optimizer_Dp.zero_grad()
        self.backward_Dp()
        self.optimizer_Dp.step()

        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.opt.stage!=1:
            self.optimizer_E.step()


    def get_current_errors(self):
        return OrderedDict([('E_t', self.loss_t),
                            ('E_c', self.loss_c),
                            ('E', self.loss_E),
                            ('G_r', self.loss_r),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('G', self.loss_G),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])

    def get_current_visuals(self):
        ach = util.tensor2im(self.ach_img)

        ach_map = self.ach_pose.sum(1)
        ach_map[ach_map>1]=1
        ach_pose = util.tensor2im(torch.unsqueeze(ach_map,1))
        ach_mask = util.tensor2im(torch.unsqueeze(self.ach_mask,1))
        ach_fake1 = util.tensor2im(self.fake_ach.data)
        ach_fake2 = util.tensor2im(self.swap_ach.data)
        ###########
        pos = util.tensor2im(self.pos_img)

        pos_map = self.pos_pose.sum(1)
        pos_map[pos_map>1]=1
        pos_pose = util.tensor2im(torch.unsqueeze(pos_map,1))
        pos_mask = util.tensor2im(torch.unsqueeze(self.pos_mask,1))

        pos_fake1 = util.tensor2im(self.fake_pos.data)
        pos_fake2 = util.tensor2im(self.swap_pos.data)

        return OrderedDict([('ach', ach), ('ach_pose', ach_pose), ('ach_mask', ach_mask), ('ach_fake1', ach_fake1), ('ach_fake2', ach_fake2),
                            ('pos', pos), ('pos_pose', pos_pose), ('pos_mask', pos_mask), ('pos_fake1', pos_fake1), ('pos_fake2', pos_fake2)])

    def save(self, epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
