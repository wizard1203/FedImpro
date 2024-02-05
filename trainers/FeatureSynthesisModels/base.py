import copy
import logging
import time
import os
import sys
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import platform


class FeatureSynthesis(nn.Module):
    def __init__(self, args, device='cpu'):
        super(FeatureSynthesis, self).__init__()

        self.args = args
        self.device = device
        self.num_classes = self.args.num_classes

        self.max_epochs = self.args.max_epochs
        self.predefined_number_per_class = 5000
        self.forward_count = 0


    def move_to_gpu(self, device):
        self.model.to(self.device)


    def move_to_cpu(self):
        self.model.to("cpu")




    @abstractmethod
    def update(self, progress, feat=None, labels=None, fake_feat=None, fake_labels=None):
        pass



    @abstractmethod
    def sample(self, x=None, labels=None):
        fake_features = None
        return fake_features



    @abstractmethod
    def initial_model_params(self, feat, feat_length=None):
        pass



    @abstractmethod
    def get_model_params(self, DP_degree=None):
        model_params = {}
        return model_params



    @abstractmethod
    def set_model_params(self, model_parameters):
        pass







































