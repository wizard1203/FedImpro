import copy
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import platform

from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from utils.data_utils import (
    get_data,
    filter_parameters,
    mean_std_online_estimate,
    retrieve_mean_std,
    get_tensors_norm,
    average_named_params,
    idv_average_named_params,
    get_name_params_div,
    get_name_params_sum,
    get_name_params_difference,
    get_name_params_difference_norm,
    get_name_params_difference_abs,
    get_named_tensors_rotation,
    calculate_metric_for_named_tensors,
    get_diff_tensor_norm,
    get_tensor_rotation,
    calculate_metric_for_whole_model,
    calc_client_divergence,
    check_device,
    scan_model_with_depth,
    average_tensors,
    add_gaussian_noise_named_tensors
)

from utils.model_utils import (
    freeze_by_names,
    unfreeze_by_names,
    freeze_by_depth,
    unfreeze_by_depth,
    get_actual_layer_names
)
from lr_scheduler.build import create_scheduler

from loss_fn.losses import SupConLoss, Distance_loss, align_feature_loss, MMD_loss
from loss_fn.ot import sinkhorn_loss_joint_IPOT

from trainers.freezer_custom import Freezer_Custom


from .FeatureSynthesisModels.build import create_synthesis





class FeatureGenerator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(FeatureGenerator, self).__init__()
        self.args = args
        self.device = device
        self.num_classes = self.args.num_classes

        self.max_epochs = self.args.max_epochs

        if self.args.dataset == 'cifar10': 
            self.predefined_number_per_class = 5000
        elif self.args.dataset == 'fmnist': 
            self.predefined_number_per_class = 6000
        elif self.args.dataset == 'SVHN': 
            self.predefined_number_per_class = 7000
        elif self.args.dataset == 'cifar100': 
            self.predefined_number_per_class = 600
        elif self.args.dataset == 'femnist': 
            self.predefined_number_per_class = 10000
        else:
            raise NotImplementedError


        self.forward_count = 0

        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis = create_synthesis(args, args.fed_split_feature_synthesis, device)
        else:
            raise NotImplementedError


    def update(self, progress, feat=None, labels=None, fake_feat=None, fake_labels=None):
        decode_error = 0.0
        if self.args.fed_split == "FeatureSynthesisLabel":
            decode_error = self.feature_synthesis.update(progress, feat, labels, fake_feat, fake_labels)
        else:
            raise NotImplementedError
        return decode_error


    def move_to_gpu(self, device):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.move_to_gpu(device)
        else:
            raise NotImplementedError


    def move_to_cpu(self):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.move_to_cpu()
        else:
            raise NotImplementedError


    def sample(self, x=None, labels=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            align_features, align_labels = self.feature_synthesis.sample(x, labels)
            return align_features, align_labels
        else:
            raise NotImplementedError


    def initial_model_params(self, feat, feat_length=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.initial_model_params(feat, feat_length)
        else:
            raise NotImplementedError


    def get_model_params(self, DP_degree=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            return self.feature_synthesis.get_model_params(DP_degree=DP_degree)
        else:
            raise NotImplementedError
        return model_params


    def set_model_params(self, model_parameters):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.set_model_params(model_parameters)
            # self.model.load_state_dict(model_parameters)
        else:
            raise NotImplementedError


    def __(self):
        if self.args.fed_split == "FeatureSynthesisLabel":
            pass
        else:
            raise NotImplementedError




class Spliter(object):
    """
        Responsible to implement split Training.
        There maybe some history information need to be memorized.
        It may need a self opimizer, due to some self parameters.
    """
    def __init__(self, args, device, model, optimizer, **kwargs):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer

        self.layer_generator_dict = {}

        self.num_classes = self.args.num_classes
        self.device = device

        self.freezer = Freezer_Custom(self.args, self.model, self.optimizer)
        self.freezer.initial_freezer(self.args.max_epochs)

        if self.args.fed_split == "FeatureSynthesisLabel":
            for i in range(self.args.fed_split_module_num):
                self.layer_generator_dict[i] = FeatureGenerator(self.args, self.device)
            self.distance_loss = Distance_loss("SupCon", device=self.device)
        else:
            raise NotImplementedError

        # self.optimizer = torch.optim.SGD(parameters_to_optim,
        #     lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        # self.lr_scheduler = create_scheduler(args, self.optimizer, **kwargs)

        if self.args.fed_split_feature_weight_sched == "default":
            self.args.base_fake_feature_weight = 1.0
        elif self.args.fed_split_feature_weight_sched == "gradual":
            self.args.base_fake_feature_weight = 1.0
        elif self.args.fed_split_feature_weight_sched == "gradual2half":
            self.args.base_fake_feature_weight = 1.0
        elif self.args.fed_split_feature_weight_sched == "gradualhalfhalf":
            self.args.base_fake_feature_weight = 0.5
        elif self.args.fed_split_feature_weight_sched == "milestone":
            self.args.base_fake_feature_weight = 1.0
        else:
            raise NotImplementedError

        if self.args.fed_split_hidden_loss is None:
            self.split_hidden_loss = None
        elif self.args.fed_split_hidden_loss == "SimCLR":
            # self.split_hidden_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
            # self.split_hidden_loss = SupConLoss(contrast_mode='all', base_temperature=1.0, device=self.device)
            self.split_hidden_loss = SupConLoss(contrast_mode='all', base_temperature=self.args.fed_split_CL_b_temperature, device=self.device)
        else:
            raise NotImplementedError

    def move_to_gpu(self, device):
        for layer, layer_generator in self.layer_generator_dict.items():
            if layer_generator is not None:
                layer_generator.move_to_gpu(device)


    def move_to_cpu(self):
        for layer, layer_generator in self.layer_generator_dict.items():
            if layer_generator is not None:
                layer_generator.move_to_cpu()


    def get_layer_generator_dict(self, DP_degree=None):
        # layer_generator_dict = copy.deepcopy(self.layer_generator_dict)
        layer_generator_dict = {}
        for layer, layer_generator in self.layer_generator_dict.items():
            layer_generator_dict[layer] = layer_generator.get_model_params(DP_degree=DP_degree)
        return layer_generator_dict


    def set_layer_generator_dict(self, layer_generator_dict):
        # for i, feature_mean in enumerate(layer_generator_dict):
        # self.layer_generator_dict[i].data = copy.deepcopy(feature_mean.data)
        for layer, layer_generator in layer_generator_dict.items():
            self.layer_generator_dict[layer].set_model_params(layer_generator)


    def average_layer_generator_dict(self, training_num, sample_num_dict, local_cls_weight_list_dict, multiple_layer_generator_dict):
        """
            Here we have a dict of dict of dict:
            multiple_layer_generator_dict:
                Client: layer_generator_dict
                    layer: layer_generator:
                        name: params
        """
        with torch.no_grad():
            if self.args.fed_split_std_mode == "update":
                avg_layer_generator_dict = {}
                all_client_list = list(local_cls_weight_list_dict.keys())
                selected_client_list =  list(sample_num_dict.keys())
                selected_weights = []
                for i, client in enumerate(selected_client_list):
                    selected_weights.append(sample_num_dict[client] / training_num)

                generator_name_list = list(list(multiple_layer_generator_dict.values())[0].keys())
                # logging.info(f"multiple_layer_generator_dict:{multiple_layer_generator_dict}")

                if self.args.fed_split_estimate_mode == "all" and self.args.fed_split_estimate_weight == "label_size":
                    previous_layer_generator_dict = self.get_layer_generator_dict()
                    for generator_name in generator_name_list:
                        avg_layer_generator_dict[generator_name] = {}
                        all_client_mean_list = []
                        for client in all_client_list:
                            if client in multiple_layer_generator_dict:
                                all_client_mean_list.append(multiple_layer_generator_dict[client][generator_name]["mean"])
                            else:
                                all_client_mean_list.append(previous_layer_generator_dict[generator_name]["mean"])
                        avg_layer_generator_dict[generator_name]["mean"] = copy.deepcopy(all_client_mean_list[0])
                        for label in range(self.args.num_classes):
                            weights = []
                            mean_label_list = []
                            for client in all_client_list:
                                weights.append(local_cls_weight_list_dict[client][label])
                                mean_label_list.append(all_client_mean_list[client][label])
                            avg_layer_generator_dict[generator_name]["mean"][label] = average_tensors(mean_label_list, weights, inplace=False)

                        if self.args.fed_split_std_mode == "update":
                            all_client_std_list = []
                            for client in all_client_list:
                                if client in multiple_layer_generator_dict:
                                    all_client_std_list.append(multiple_layer_generator_dict[client][generator_name]["std"]**2)
                                else:
                                    all_client_std_list.append(previous_layer_generator_dict[generator_name]["std"]**2)
                            # for label in range(self.args.num_classes):
                            #     weights = []
                            #     mean_label_list = []
                            #     for client in all_client_list:
                            #         weights.append(local_cls_weight_list_dict[client][label])
                            #         mean_label_list.append(all_client_mean_list[client][label])
                            #     avg_layer_generator_dict[generator_name]["mean"] = average_tensors(mean_label_list, weights, inplace=False)
                            avg_std = copy.deepcopy(all_client_std_list[0])
                            for label in range(self.args.num_classes):
                                weights = []
                                std_label_list = []
                                for client in all_client_list:
                                    weights.append(local_cls_weight_list_dict[client][label])
                                    std_label_list.append(all_client_std_list[client][label])
                                avg_std[label] = average_tensors(std_label_list, weights, inplace=False)

                            for label in range(self.args.num_classes):
                                var_mean = None
                                for client in all_client_list:
                                    var_mean = local_cls_weight_list_dict[client][label] * \
                                        (all_client_mean_list[client][label].to(avg_std[label].device) - \
                                            avg_layer_generator_dict[generator_name]["mean"][label].to(avg_std[label].device))**2
                                    avg_std[label] += var_mean
                        avg_layer_generator_dict[generator_name]["std"] = avg_std.sqrt()
                elif self.args.fed_split_estimate_mode == "selected" and self.args.fed_split_estimate_weight == "uniform":
                    avg_layer_generator_dict = {}
                    for generator_name in generator_name_list:
                        client_generator_list = []
                        weights = []
                        for client in selected_client_list:
                            client_generator_list.append(multiple_layer_generator_dict[client][generator_name])
                            weights.append(1/len(selected_client_list))
                            # logging.info(f"multiple_layer_generator_dict[{client}][{generator_name}]:{multiple_layer_generator_dict[client][generator_name]}")
                        avg_layer_generator_dict[generator_name] = average_named_params(client_generator_list, weights, inplace=False)
                else:
                    raise NotImplementedError

            elif self.args.fed_split_std_mode == "default":
                avg_layer_generator_dict = {}

                client_list = list(multiple_layer_generator_dict.keys())
                generator_name_list = list(multiple_layer_generator_dict[client_list[0]].keys())
                # logging.info(f"multiple_layer_generator_dict:{multiple_layer_generator_dict}")
                for generator_name in generator_name_list:
                    client_generator_list = []
                    weights = []
                    for client in client_list:
                        client_generator_list.append(multiple_layer_generator_dict[client][generator_name])
                        weights.append(1/len(client_list))
                        # logging.info(f"multiple_layer_generator_dict[{client}][{generator_name}]:{multiple_layer_generator_dict[client][generator_name]}")
                    avg_layer_generator_dict[generator_name] = average_named_params(client_generator_list, weights, inplace=False)
        return avg_layer_generator_dict


    def server_update_layer_generator_dict(self, training_num, sample_num_dict, local_cls_weight_list_dict, multiple_layer_generator_dict):
        # local_cls_weight_list_dict
        # multiple_layer_generator_dict
        avg_layer_generator_dict = self.average_layer_generator_dict(
            training_num, sample_num_dict, local_cls_weight_list_dict, multiple_layer_generator_dict)
        self.set_layer_generator_dict(avg_layer_generator_dict)


    def generate_layer_generator_dict(self, model, x, labels=None):
        # Server is responsible to call this function.
        model.eval()
        model.to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            output, feat_list, hidden_loss, loss = model.split_forward(x, labels, progress=0)
            for i, feat in enumerate(feat_list):
                logging.info(f"Initializing {i}-th feature generator......")
                logging.info(f"feature shape: {feat.shape} ......")
                logging.info(f"model.feat_length_list[{i}]: {model.feat_length_list[i]} ......")
                self.layer_generator_dict[i].initial_model_params(feat.to("cpu"), model.feat_length_list[i])
        model.to("cpu")


    def freeze_layers(self, progress):
        if self.args.fed_split_freeze_front and (not (progress < self.args.fed_split_freeze_begin_epochs)):
            if self.args.model in ["resnet18_v2", "resnet34_v2", "resnet50_v2"]:
                layer_sub_name_list = ["conv1", "bn1"]
                for layer_sub_name in layer_sub_name_list:
                    switch_index_list = self.freezer.switch_choosen_layers(layer_sub_name, "fullname", self.optimizer, freeze=True)
                    logging.info(f"{switch_index_list} are frozen!!!!!! ")
                layer_sub_name_list = []
                if self.args.fed_split_module_num > 1:
                    raise NotImplementedError
                else:
                    layer = int(self.args.fed_split_module_choose.split("layer")[1])
                    for layer in range(1, layer+1):
                        layer_sub_name_list.append(f"layer{layer}")
                    layer_sub_name_list.append(f"layer{layer}")
                for layer_sub_name in layer_sub_name_list:
                    switch_index_list = self.freezer.switch_choosen_layers(layer_sub_name, "subname", self.optimizer, freeze=True)
                    logging.info(f"{switch_index_list} are frozen!!!!!! ")
            else:
                raise NotImplementedError


    def split_train(self, model, x, labels, loss_func, progress=0):

        real_batch_size = x.size(0)
        fake_labels_list = []

        split_loss = 0.0
        hidden_loss = 0.0
        decode_error = 0
        if self.args.fed_split == "FeatureSynthesisLabel":
            # for i, feat in enumerate(feat_list):
            fake_features_list = []
            fake_labels_list = []
            for i in range(self.args.fed_split_module_num):
                fake_features, fake_labels = self.layer_generator_dict[i].sample(x, labels)
                # logging.info(f"fake_features.shape: {fake_features.shape}")
                fake_features_list.append(fake_features.detach())
                fake_labels_list.append(fake_labels.detach())

            if self.args.fed_split_hidden_loss is not None:
                model.split_hidden_loss = self.split_hidden_loss

            output, feat_list, hidden_loss, loss = model.split_forward(x, labels, fake_features_list, fake_labels_list, progress=progress)
            # split_loss += hidden_loss
            for i, feat in enumerate(feat_list):
                cache_feature_means = copy.deepcopy(self.layer_generator_dict[i].feature_synthesis.feature_means)
                if self.args.fed_split_forward_decouple:
                    if i == 0:
                        decode_error = self.layer_generator_dict[i].update(progress, feat.detach(), labels.detach(), fake_features_list[i], fake_labels_list[i])
                    else:
                        decode_error = self.layer_generator_dict[i].update(progress, feat.detach(), fake_labels_list[i].detach(), fake_features_list[i], fake_labels_list[i])
                else:
                    decode_error = self.layer_generator_dict[i].update(progress, feat.detach(), labels.detach())

                if self.args.fed_split_constrain_feat:
                    # assert 
                    loss = self.distance_loss(feat, fake_features_list[i], labels, fake_labels_list[i])
                else:
                    loss = torch.tensor(0.0)
                split_loss += loss

                diff = self.layer_generator_dict[i].feature_synthesis.feature_means - cache_feature_means
                logging.info(f"After Updating, self.feature_means [{i}] change norm: {diff.norm(p=2)}")
        else:
            raise NotImplementedError

        if self.args.fed_split == "FeatureSynthesisLabel":
            if self.args.fed_split_forward_decouple:
                # Use decouple, there will be only real feature forwarded to the last layer.
                # hidden loss has been backwarded in the model.split_forward().
                # loss = loss_func(fake_features_list[-1], fake_labels_list[-1])
                loss = loss_func(output, fake_labels_list[-1])
                total_loss = split_loss*self.args.fed_split_loss_weight + loss
                total_loss.backward()
                split_loss_value = split_loss.item()
                loss = loss.item()
                # output = fake_features_list[-1]
            else:
                # Use detach or not use, there will be real feature and fake features forwarded to the last layer.
                all_labels = [labels]
                for fake_labels in fake_labels_list:
                    all_labels.append(fake_labels)
                all_labels = torch.cat(all_labels, dim=0)

                if self.args.fed_split_feature_weight_sched == "default":
                    loss = loss_func(output, all_labels)
                    total_loss = split_loss*self.args.fed_split_loss_weight + loss
                    total_loss.backward()

                elif self.args.fed_split_feature_weight_sched == "gradual":
                    # fake_weight = self.args.base_fake_feature_weight * (progress / self.args.max_epochs)
                    fake_weight = self.args.base_fake_feature_weight * (progress / self.args.fed_split_feature_weight_max_epochs)
                    logging.info(f"In fed split, using gradual split feature weight, progress:{progress}, fake_weight:{fake_weight}")
                    loss_real = loss_func(output[:real_batch_size], all_labels[:real_batch_size])
                    loss_fake = loss_func(output[real_batch_size:], all_labels[real_batch_size:])
                    loss = loss_real * (1 - fake_weight) + loss_fake * fake_weight
                    total_loss = split_loss*self.args.fed_split_loss_weight + loss
                    total_loss.backward()
                elif self.args.fed_split_feature_weight_sched == "gradual2half":
                    fake_weight = self.args.base_fake_feature_weight * (progress / self.args.fed_split_feature_weight_max_epochs)
                    logging.info(f"In fed split, using gradual split feature weight, progress:{progress}, fake_weight:{fake_weight}")
                    loss_real = loss_func(output[:real_batch_size], all_labels[:real_batch_size])
                    loss_fake = loss_func(output[real_batch_size:], all_labels[real_batch_size:])
                    loss = loss_real * 1 + loss_fake * fake_weight
                    total_loss = split_loss*self.args.fed_split_loss_weight + loss
                    total_loss.backward()

                elif self.args.fed_split_feature_weight_sched == "gradualhalfhalf":
                    # Now, real weight is 1 -> 0.5, fake weight is 0 -> 0.5.
                    fake_weight = self.args.base_fake_feature_weight * (progress / self.args.fed_split_feature_weight_max_epochs)
                    logging.info(f"In fed split, using gradual split feature weight, progress:{progress}, fake_weight:{fake_weight}")
                    loss_real = loss_func(output[:real_batch_size], all_labels[:real_batch_size])
                    loss_fake = loss_func(output[real_batch_size:], all_labels[real_batch_size:])
                    loss = loss_real * (1 - fake_weight) + loss_fake * fake_weight
                    total_loss = split_loss*self.args.fed_split_loss_weight + loss
                    total_loss.backward()

                elif self.args.fed_split_feature_weight_sched == "milestone":
                    if progress < self.args.fed_split_feature_weight_max_epochs:
                        logging.info(f"In fed split, using milestone split feature weight, progress:{progress}")
                        loss_real = loss_func(output[:real_batch_size], all_labels[:real_batch_size])
                        loss = loss_real
                    else:
                        loss_fake = loss_func(output[real_batch_size:], all_labels[real_batch_size:])
                        loss = loss_fake
                    total_loss = split_loss*self.args.fed_split_loss_weight + loss
                    total_loss.backward()

                else:
                    raise NotImplementedError

                split_loss_value = split_loss.item()
                loss = loss.item()
                #  Only take outputs of the real data for measuring training accuracy.
                output = output[:real_batch_size]
        else:
            loss = loss_func(output, labels)
            total_loss = split_loss*self.args.fed_split_loss_weight + loss
            total_loss.backward()
            split_loss_value = split_loss.item()
            loss = loss.item()
        # np.isnan(total_loss.item())
        # return output, feat_list, split_loss.item(), loss.item()
        return output, feat_list, fake_labels_list, split_loss_value, hidden_loss, decode_error, loss
























