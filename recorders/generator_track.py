import os
import logging

import torch

from configs.chooses import EPOCH, ITERATION

class generator_tracker(object):

    def __init__(self, args=None):
        self.things_to_track = ["generator_track"]

    def check_config(self, args, **kwargs):
        return True

    def generate_record(self, args, **kwargs):
        """ Here args means the overall args, not the *args """
        info_dict = {}

        if "split_loss_value" in kwargs:
            info_dict["Split-Loss"] = kwargs["split_loss_value"]

        if "split_hidden_loss" in kwargs:
            info_dict["Split-Hidden-Loss"] = kwargs["split_hidden_loss"]

        if "split_decode_error" in kwargs:
            info_dict["SplitDecode-Loss"] = kwargs["split_decode_error"]

        logging.info('generator Losses TRACK::::   {}'.format(
            info_dict
        ))
        return info_dict

    def get_things_to_track(self):
        return self.things_to_track














