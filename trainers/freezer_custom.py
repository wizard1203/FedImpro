import copy
import logging
import time

import torch
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
    scan_model_with_depth
)

from utils.model_utils import (
    freeze_by_names,
    unfreeze_by_names,
    freeze_by_depth,
    unfreeze_by_depth,
    get_actual_layer_names
)

from utils.tensor_buffer import (
    TensorBuffer
)


class _place_holder(object):
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.grad = None


FREEZED = "Freezed"
UNFREEZED = "UnFreezed"

class Freezer_Custom(object):
    """
        Responsible to implement freezing.
        There maybe some history information need to be memorized.
    """
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.named_modules = dict(model.named_modules())
        self.optimizer = optimizer
        self.placeholder = _place_holder(args, "Freezed")
        self.freeze_status = {}


    def initial_freezer(self, max_progress):
        """Remember to Call this function before training"""
        self.max_progress = max_progress
        # self.named_parameters, self.named_modules, self.depth_dict = \
        #     scan_model_with_depth(self.model)
        self.depth_dict = {}
        # depth count begins from 1
        self.max_depth = 1
        self.layer_names = []
        for param_group in self.optimizer.param_groups:
            # logging.info(f"param_name: {param_group['param_name']}")
            # logging.info(f"layer_name: {param_group['layer_name']}, ")
            # logging.info(f"depth: {param_group['depth']}")
            self.depth_dict[param_group["param_name"]] = param_group["depth"]
            self.max_depth = max(self.max_depth, param_group["depth"])
            self.freeze_status[param_group["param_name"]] = UNFREEZED
            self.layer_names.append(param_group["param_name"])
            # self.freeze_status[param_group["depth"]] = UNFREEZED
        logging.info(f"Depth Dict: {self.depth_dict}")


    def switch_choosen_layers(self, layer_sub_name, name_match, optimizer, freeze=True, freeze_bn=False):
        switch_index_list = self.filter_and_process(layer_sub_name, name_match, optimizer, freeze, freeze_bn)
        return switch_index_list


    # param_groups.append({'params': param, "param_name": name, "layer_name": module_name, "depth": layers_depth[module_name]})
    # def filter_and_process(self, freeze_layers_list, optimizer, freeze=True):
    def filter_and_process(self, layer_sub_name, name_match, optimizer, freeze=True, freeze_bn=False):
        switch_index_list = []
        if freeze:
            for param_group in optimizer.param_groups:
                # Note that the param_group stores each param, one layer may have more than 2 params,
                # Thus the remove index does not represents the layer index.
                depth = param_group["depth"]
                layer_name = param_group["layer_name"]
                param_name = param_group["param_name"]
                # logging.info("")
                if (layer_sub_name == layer_name and name_match == "fullname") or \
                    (layer_sub_name in layer_name and name_match == "subname") :
                    # This remove operation maybe changed with the version of Pytorch.
                    # optimizer.state.pop[param_group['params']]
                    param_group['params'] = [self.placeholder]
                    # optimizer.state[self.placeholder] = self.placeholder
                    # optimizer.state[id([])] = None
                    # switch_index_list.append(param_group)
                    switch_index_list.append(layer_name)
                    module = self.named_modules[layer_name]
                    for param in module.parameters():
                        param.requires_grad = not freeze
                    if "Norm" in type(module).__name__ and freeze_bn:
                        logging.info(f"detech Norm Layer {layer_name}, Freezing..........")
                        module.eval()
                        module.track_running_stats = False
                    self.freeze_status[param_group["param_name"]] = FREEZED
                    # logging.info(f'Detect layer, Freezed Layer: {layer_name},, sub_name:{layer_sub_name}, param name: {param_group["param_name"]} \
                    #     Status: {self.freeze_status[param_group["param_name"]]}')
                else:
                    pass
                    # logging.info(f'Not Detect layer--{layer_name}, sub_name:{layer_sub_name},Freezed Layer: {param_group["layer_name"]}, param name: {param_group["param_name"]}\
                    #     Status: {self.freeze_status[param_group["param_name"]]}')
        else:
            raise NotImplementedError
        return switch_index_list


    def freeze_layers(self, model_name, freeze_backbone_layers, freeze_bn=False):
        if model_name in ["resnet18_v2", "resnet34_v2", "resnet50_v2"]:
            part = freeze_backbone_layers.split("-")[0]
            if part == "Before":
                pass
            elif part == "After":
                raise NotImplementedError
            else:
                raise NotImplementedError

            layer_sub_name_list = ["conv1", "bn1"]
            for layer_sub_name in layer_sub_name_list:
                switch_index_list = self.switch_choosen_layers(layer_sub_name, "fullname", self.optimizer, freeze=True, freeze_bn=freeze_bn)
                logging.info(f"{switch_index_list} are frozen!!!!!! ")
            layer_sub_name_list = []

            submodel_name = freeze_backbone_layers.split("-")[1]
            layer = int(submodel_name[-1])

            for layer in range(1, layer+1):
                layer_sub_name_list.append(f"layer{layer}")
            layer_sub_name_list.append(f"layer{layer}")
            for layer_sub_name in layer_sub_name_list:
                switch_index_list = self.switch_choosen_layers(layer_sub_name, "subname", self.optimizer, freeze=True, freeze_bn=freeze_bn)
                logging.info(f"{switch_index_list} are frozen!!!!!! ")
        else:
            raise NotImplementedError
























