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

from .GaussianSynthesisLabel import GaussianSynthesisLabel

def create_synthesis(args, model_name, device, **kwargs):
    logging.info("create synthesis...... model_name = %s" %
                (model_name))
    model = None
    logging.info(f"model name: {model_name}")

    if model_name == "GaussianSynthesisLabel":
        model = GaussianSynthesisLabel(args, device)
    else:
        raise NotImplementedError

    return model

