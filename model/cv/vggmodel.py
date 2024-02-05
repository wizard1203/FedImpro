import math
from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    # def __init__(self, features, input_channels=3, output_dim=10, args=None, device=None):
    def __init__(self, layers, input_channels=3, output_dim=10, args=None, device=None):
        super(VGG, self).__init__()
        self.args = args
        # self.image_size = image_size
        self.device = device

        self.feat_length_list = [None for i in range(self.args.fed_split_module_num)]

        # self.features = features
        # self.layers = layers
        self.num_feature_layers = len(layers)
        for index in range(self.num_feature_layers):
            exec('self.layer' + str(index) + '= layers[index]')

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def process_split_feature(self, real_batch_size, real_feat, real_label, real_feat_list, hidden_loss=None,
                            fake_feat=None, fake_label=None, progress=None):
        if self.args.fed_split_hidden_detach or self.args.fed_split_forward_decouple:
            if progress < self.args.fed_split_forward_detach_decouple_begin_epochs:
                if self.training and hidden_loss is not None:
                    hidden_loss.backward(retain_graph=True)
            else:
                if self.training and hidden_loss is not None:
                    hidden_loss.backward()
                logging.info(f"in progress: {progress} real_feat is detached!!!!!!!!!!!!!!!!!!!!!!!!!!")
                real_feat = real_feat.detach()
        elif hidden_loss is not None:
            if self.training:
                hidden_loss.backward(retain_graph=True)
        else:
            pass

        if fake_feat is not None:
            # logging.info(f"fake_feat.shape: {fake_feat.shape}")

            if self.args.fed_split_forward_decouple:
                next_feat = fake_feat.reshape(
                    -1, real_feat.shape[1], real_feat.shape[2], real_feat.shape[3])
                real_feat_list.append(real_feat.view(real_feat.size(0), -1) * 1.0)
                next_label = fake_label
            else:
                next_feat = torch.cat([real_feat, fake_feat.reshape(
                    -1, real_feat.shape[1], real_feat.shape[2], real_feat.shape[3])], dim=0)
                real_feat_list.append((real_feat.view(real_feat.size(0), -1) * 1.0)[:real_batch_size])
                next_label = torch.cat([real_label, fake_label], dim=0)
            # feat_list.append(feat[:real_feat.size(0) - hidden_features[extra_feat_index].size(0)])
            # logging.info(f"feat[:real_feat.size(0) - hidden_features[extra_feat_index].size(0)].shape: \
            #     {feat[:real_feat.size(0) - hidden_features[extra_feat_index].size(0)].shape}")
        else:
            # assert (not self.args.fed_split_forward_decouple)
            # next_feat = real_feat.view(real_feat.size(0), -1) * 1.0
            next_feat = real_feat
            real_feat_list.append(real_feat.view(real_feat.size(0), -1) * 1.0)
            next_label = real_label
        return next_feat, next_label

    def forward(self, x):
        for index in range(self.num_feature_layers):
            x = eval('self.layer' + str(index))(x)
            # if index == 4:
        feat = x.view(x.size(0), -1) * 1.0
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        if self.args.model_out_feature:
            return out, feat
        else:
            return out


    # def forward(self, x):
    #     x = self.features(x)
    #     feat = x.view(x.size(0), -1)
    #     out = self.classifier(feat)
    #     if self.args.model_out_feature:
    #         return out, feat
    #     else:
    #         return out

    def split_forward(self, x, target=None, hidden_features=None, hidden_labels=None, progress=None):
        assert self.args.fed_split is not None
        stage_i = 0
        layer_i = 0
        local_module_i = 0
        decode_img_list = []
        feat_list = []

        total_hidden_loss = torch.tensor(0.0)

        extra_feat_index = 0
        real_batch_size = x.size(0)

        next_target = deepcopy(target)

        # x = self.features(img)
        for index in range(self.num_feature_layers):
            x = eval('self.layer' + str(index))(x)
            # if index == 4:
        if hidden_features is not None:
            x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                    hidden_loss=None,
                                    fake_feat=hidden_features[local_module_i], fake_label=hidden_labels[local_module_i],
                                    progress=progress)
        else:
            x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                    hidden_loss=None,
                                    fake_feat=None, fake_label=None,
                                    progress=progress)
            self.feat_length_list[local_module_i] = x.size(-1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        loss = torch.tensor(1.0)

        return x, feat_list, total_hidden_loss.item(), loss.item()



def make_layers(cfg, input_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.append(nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]))
            else:
                layers.append(nn.Sequential(*[conv2d, nn.ReLU(inplace=True)]))
            input_channels = v
    return layers
    # return nn.Sequential(*layers)


def make_layers_old(cfg, input_channels=3, batch_norm=False):
    layers = []
    input_channels = input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            input_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], input_channels=input_channels), output_dim=output_dim,
               args=args, device=device)


def vgg11_bn(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], input_channels=input_channels, batch_norm=True), output_dim=output_dim,
               args=args, device=device)


def vgg13(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], input_channels=input_channels), output_dim=output_dim,
               args=args, device=device)


def vgg13_bn(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], input_channels=input_channels, batch_norm=True), output_dim=output_dim,
               args=args, device=device)


def vgg16(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], input_channels=input_channels), output_dim=output_dim,
               args=args, device=device)


def vgg16_bn(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], input_channels=input_channels, batch_norm=True), output_dim=output_dim,
               args=args, device=device)


def vgg19(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], input_channels=input_channels), output_dim=output_dim,
               args=args, device=device)


def vgg19_bn(input_channels=3, output_dim=10, args=None, device=None):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], input_channels=input_channels, batch_norm=True), output_dim=output_dim,
               args=args, device=device)


