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

from .base import FeatureSynthesis

from utils.data_utils import (
    add_gaussian_noise_named_tensors
)


class GaussianSynthesisLabel(FeatureSynthesis):
    def __init__(self, args, device='cpu'):
        # super(GaussianSynthesisLabel, self).__init__(args, device)
        super().__init__(args, device)

        self.feature_means = None
        self.feature_std = None


    def move_to_gpu(self, device):
        self.feature_means = self.feature_means.to(device)
        self.feature_std = self.feature_std.to(device)


    def move_to_cpu(self):
        self.feature_means = self.feature_means.to("cpu")
        self.feature_std = self.feature_std.to("cpu")


    def update(self, progress, feat=None, labels=None, fake_feat=None, fake_labels=None):
        dif_error = 0.0
        with torch.no_grad():
            for label in range(self.num_classes):
                if torch.any(labels==label):

                    feat_mean_label = feat[labels==label].mean(dim=0)
                    feat_std_label = feat[labels==label].std(dim=0)
                    dif_error += (feat_mean_label - self.feature_means[label]).norm(p=2)
                    weight = min(1.0, feat_mean_label.size(0)/self.predefined_number_per_class)

                    self.feature_means[label] = self.feature_means[label]*(1 - weight) \
                        + feat_mean_label*weight
                    if self.args.fed_split_std_mode == "update" and not torch.any(torch.isnan(feat_std_label)):
                        self.feature_std[label] = self.feature_std[label]*(1 - weight) \
                            + feat_std_label*weight
                else:
                    pass
                if self.args.fed_split_std_mode == "update":
                    self.feature_std[self.feature_std > self.args.fed_split_noise_std] = self.args.fed_split_noise_std
            # logging.info(f"self.feature_std is updated as :{self.feature_std}")
            logging.info(f"Average of self.feature_std is :{self.feature_std.mean()}")
        return dif_error.item()


    def sample(self, x=None, labels=None):
        repeat_times = x.size(0) // self.num_classes + 1
        fake_labels = torch.tensor(list(range(0, self.num_classes))*repeat_times).to(self.device)
        fake_features = self.feature_means.repeat(repeat_times, 1).to(self.device)
        if self.args.fed_split_std_mode == "update":
            fake_std = self.feature_std.repeat(repeat_times, 1).to(self.device)
        else:
            fake_std = self.feature_std
        fake_features = torch.normal(mean=fake_features, std=fake_std)
        # logging.info(f"GaussianSynthesisLabel sample fake feature shape:{fake_features.shape}")
        return fake_features, fake_labels


    def initial_model_params(self, feat, feat_length=None):
        self.feature_means = torch.rand(self.args.num_classes, feat.shape[1])
        if self.args.fed_split_std_mode == "update":
            self.feature_std = torch.ones([self.args.num_classes, feat.shape[1]])*self.args.fed_split_noise_std
        else:
            self.feature_std = torch.tensor(self.args.fed_split_noise_std)


    def get_model_params(self, DP_degree=None):
        model_params = {"mean": self.feature_means.data, "std": self.feature_std.data}
        if DP_degree is not None:
            add_gaussian_noise_named_tensors(model_params, std=DP_degree)
        return model_params


    def set_model_params(self, model_parameters):
        self.feature_means.data = model_parameters["mean"]
        self.feature_std.data = model_parameters["std"]














