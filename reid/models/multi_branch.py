from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .resnet import BasicBlock, Bottleneck, ResNet
import torchvision

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class SingleNet(nn.Module):
    in_planes = 2048
    def __init__(self, model_name, num_classes, pretraind=True, use_bn=True, test_bn=True, last_stride=2):
        super(SingleNet, self).__init__()

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base_model = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            if pretraind:
                origin = torchvision.models.resnet18(pretrained=True).state_dict()
                    
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base_model = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            if pretraind:
                origin = torchvision.models.resnet34(pretrained=True).state_dict()

        elif model_name == 'resnet50':
            self.base_model = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            if pretraind:
                origin = torchvision.models.resnet50(pretrained=True).state_dict()

        elif model_name == 'resnet101':
            self.base_model = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            if pretraind:
                origin = torchvision.models.resnet101(pretrained=True).state_dict()

        elif model_name == 'resnet152':
            self.base_model = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            if pretraind:
                origin = torchvision.models.resnet152(pretrained=True).state_dict()
        else:
            raise ValueError('unrecognized base model')

        if pretraind:
            for key in origin:
                if key.find('fc') != -1: continue
                self.base_model.state_dict()[key].copy_(origin[key])
            del origin
            print('load model: %s' %  model_name)
        else:
            self.base_model.random_init()


        self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.test_bn = test_bn

        if self.use_bn == False:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base_model(x)) 
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.use_bn == False:
            feat = global_feat
        else:
            feat = self.bottleneck(global_feat)

        if self.training:
            pred = self.classifier(feat)
            return global_feat, pred
        else:
            if self.test_bn:
                return feat
            else:
                return global_feat


class SiameseNet(nn.Module):
    in_planes = 2048
    def __init__(self, model_name, num_classes, pretraind=True, use_bn=True, test_bn=True, last_stride=2):
        super(SiameseNet, self).__init__()

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base_model = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            if pretraind:
                origin = torchvision.models.resnet18(pretrained=True).state_dict()
                    
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base_model = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            if pretraind:
                origin = torchvision.models.resnet34(pretrained=True).state_dict()

        elif model_name == 'resnet50':
            self.base_model = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            if pretraind:
                origin = torchvision.models.resnet50(pretrained=True).state_dict()

        elif model_name == 'resnet101':
            self.base_model = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            if pretraind:
                origin = torchvision.models.resnet101(pretrained=True).state_dict()

        elif model_name == 'resnet152':
            self.base_model = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            if pretraind:
                origin = torchvision.models.resnet152(pretrained=True).state_dict()
        else:
            raise ValueError('unrecognized base model')


        if pretraind:
            for key in origin:
                if key.find('fc') != -1: continue
                self.base_model.state_dict()[key].copy_(origin[key])
            del origin
            print('load model: %s' % model_name)
        else:
            self.base_model.random_init()

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self.test_bn = test_bn

        if self.use_bn == False:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base_model(x)) 
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.use_bn == False:
            feat = global_feat
        else:
            feat = self.bottleneck(global_feat)

        if self.training:
            bs = global_feat.size(0)

            gf1 = global_feat[:bs//2]
            gf2 = global_feat[bs//2:]
            f1 = feat[:bs//2]
            f2 = feat[bs//2:]
            f = (f1-f2).pow(2)
            pred = self.classifier(f)
            return gf1, gf2, pred
        else:
            if self.test_bn:
                return feat
            else:
                return global_feat
