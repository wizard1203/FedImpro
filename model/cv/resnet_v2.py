'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, args=None, image_size=32, model_input_channels=3, device=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.args = args
        self.image_size = image_size
        self.device = device

        self.feat_length_list = [None for i in range(self.args.fed_split_module_num)]

        self.conv1 = nn.Conv2d(model_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers_name_map = {
            "classifier": "linear"
        }

        inplanes = [64, 64, 128, 256, 512]
        inplanes = [ inplane * block.expansion for inplane in inplanes]
        logging.info(inplanes)


        if self.args.fed_split is not None and self.args.fed_split_hidden_loss == "SimCLR":
            if self.args.fed_split_module_num > 1:
                for layer in range(self.args.fed_split_module_num):
                    exec('self.CL_encoder_' + str(layer+1) +
                        '= AuxClassifier(inplanes=inplanes[layer], net_config=self.args.fed_split_CL_encoder_config, '
                        'loss_mode="SimCLR", class_num=num_classes, '
                        'widen=1, feature_dim=self.args.fed_split_CL_hidden_dim,'
                        'sup_con_temp=self.args.fed_split_CL_b_temperature, device=self.device)')
            else:
                layer = int(self.args.fed_split_module_choose.split("layer")[1])
                exec('self.CL_encoder_' + str(layer) +
                    '= AuxClassifier(inplanes=inplanes[layer], net_config=self.args.fed_split_CL_encoder_config, '
                    'loss_mode="SimCLR", class_num=num_classes, '
                    'widen=1, feature_dim=self.args.fed_split_CL_hidden_dim,'
                    'sup_con_temp=self.args.fed_split_CL_b_temperature, device=self.device)')

    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def get_hidden_loss_sched_weight(self, progress):

        if self.args.fed_split_hidden_weight_sched == "default":
            hidden_loss_weight = 1.0
        elif self.args.fed_split_hidden_weight_sched == "gradual":
            hidden_loss_weight = 1.0 * (1 - progress / self.args.fed_split_hidden_weight_max_epochs)
            logging.info(f"In fed split, using gradual split feature weight, progress:{progress}, hidden_loss_weight:{hidden_loss_weight}")
        elif self.args.fed_split_hidden_weight_sched == "milestone":
            logging.info(f"In fed split, using milestone split feature weight, progress:{progress}")
            if progress < self.args.fed_split_hidden_weight_max_epochs:
                hidden_loss_weight = 1.0
            else:
                hidden_loss_weight = 0.0
        else:
            raise NotImplementedError
        return hidden_loss_weight


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


    def calculate_hidden_loss(self, feat, label, layer=1, progress=None):
        if hasattr(self, "split_hidden_loss") and self.split_hidden_loss is not None:

            hidden_loss_weight = self.get_hidden_loss_sched_weight(progress)

            if self.args.fed_split_hidden_loss == "SimCLR":
                if hidden_loss_weight > 0.0:
                    logging.info(f"In hidden loss calculationg, feat.device:{feat.device}")
                    hidden_loss = eval('self.CL_encoder_' + str(layer))(
                        feat, label, self.args.fed_split_CL_temperature) * self.args.fed_split_hidden_loss_weight * hidden_loss_weight
                else:
                    hidden_loss = None
            else:
                raise NotImplementedError
        else:
            hidden_loss = None
        return hidden_loss


    def split_forward(self, img, target=None, hidden_features=None, hidden_labels=None, progress=None):
        assert self.args.fed_split is not None
        stage_i = 0
        layer_i = 0
        local_module_i = 0

        decode_img_list = []
        feat_list = []

        total_hidden_loss = torch.tensor(0.0)

        extra_feat_index = 0
        real_batch_size = img.size(0)

        next_target = deepcopy(target)

        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        if self.args.fed_split_module_num > 3 or \
            (self.args.fed_split_module_num == 1 and self.args.fed_split_module_choose == "layer1"):
            hidden_loss = self.calculate_hidden_loss(x, next_target, layer=1, progress=progress)
            total_hidden_loss += hidden_loss.item() if hidden_loss is not None else 0.0

            if hidden_features is not None:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=hidden_features[local_module_i], fake_label=hidden_labels[local_module_i],
                                        progress=progress)
            else:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=None, fake_label=None,
                                        progress=progress)
            # logging.info(f"x.shape: {x.shape}")
            self.feat_length_list[local_module_i] = x.size(-1)
            local_module_i += 1
            extra_feat_index += 1
            # logging.debug(f"Output feat after layer 1. feat shape: {feat.shape}, out.shape: {out.shape}")

        x = self.layer2(x)
        if self.args.fed_split_module_num > 2 or \
            (self.args.fed_split_module_num == 1 and self.args.fed_split_module_choose == "layer2"):
            hidden_loss = self.calculate_hidden_loss(x, next_target, layer=2, progress=progress)
            total_hidden_loss += hidden_loss.item() if hidden_loss is not None else 0.0

            if hidden_features is not None:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=hidden_features[local_module_i], fake_label=hidden_labels[local_module_i],
                                        progress=progress)
            else:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=None, fake_label=None,
                                        progress=progress)
            # logging.info(f"x.shape: {x.shape}")
            self.feat_length_list[local_module_i] = x.size(-1)
            local_module_i += 1
            extra_feat_index += 1

        x = self.layer3(x)
        if self.args.fed_split_module_num > 1 or \
            (self.args.fed_split_module_num == 1 and self.args.fed_split_module_choose == "layer3"):
            hidden_loss = self.calculate_hidden_loss(x, next_target, layer=3, progress=progress)
            total_hidden_loss += hidden_loss.item() if hidden_loss is not None else 0.0

            if hidden_features is not None:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=hidden_features[local_module_i], fake_label=hidden_labels[local_module_i],
                                        progress=progress)
            else:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=None, fake_label=None,
                                        progress=progress)
            # logging.info(f"x.shape: {x.shape}")
            self.feat_length_list[local_module_i] = x.size(-1)
            local_module_i += 1
            extra_feat_index += 1

        x = self.layer4(x)
        x = self.avgpool(x)
        if self.args.fed_split_module_num > 1 or \
            (self.args.fed_split_module_num == 1 and self.args.fed_split_module_choose == "layer4"):
            hidden_loss = self.calculate_hidden_loss(x, next_target, layer=4, progress=progress)
            total_hidden_loss += hidden_loss.item() if hidden_loss is not None else 0.0

            if hidden_features is not None:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=hidden_features[local_module_i], fake_label=hidden_labels[local_module_i],
                                        progress=progress)
            else:
                x, next_target = self.process_split_feature(real_batch_size, real_feat=x, real_label=next_target, real_feat_list=feat_list,
                                        hidden_loss=hidden_loss,
                                        fake_feat=None, fake_label=None,
                                        progress=progress)
            # logging.info(f"x.shape: {x.shape}")
            self.feat_length_list[local_module_i] = x.size(-1)
            local_module_i += 1
            extra_feat_index += 1

        x = self.linear(x.view(x.size(0), -1))
        loss = torch.tensor(1.0)

        return x, feat_list, total_hidden_loss.item(), loss.item()


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer1":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 1. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer2(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer2":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 2. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer3(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer3":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 3. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.layer4(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "resnet-layer4":
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat after layer 4. feat shape: {feat.shape}, out.shape: {out.shape}")
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        if self.args.model_out_feature and self.args.model_out_feature_layer == "last":
            # feat = out
            feat = out.view(out.size(0), -1) * 1.0
            # logging.debug(f"Output feat before last layer. feat shape: {feat.shape}, out.shape: {out.shape}")
        out = self.linear(out.view(out.size(0), -1))

        if self.args.model_out_feature:
            return out, feat
        else:
            return out


def ResNet10(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes, args=args, **kwargs)

def ResNet18(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, args=args, **kwargs)

def ResNet34(args, num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)

def ResNet50(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, args=args, **kwargs)

def ResNet101(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, args=args, **kwargs)

def ResNet152(args, num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, args=args, **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()















