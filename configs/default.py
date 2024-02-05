from curses.ascii import FF
from logging import FATAL
import os
import random

# from .config import CfgNode as CN
from .config import CfgNode as CN

_C = CN()



# ---------------------------------------------------------------------------- #
# version control
# This is used to filter experiment results.
# ---------------------------------------------------------------------------- #
_C.version = 'v1.0'
_C.exp_mode = 'debug'    # debug or normal or ready


# ---------------------------------------------------------------------------- #
# wandb settings
# ---------------------------------------------------------------------------- #
_C.entity = None
_C.project = 'test'
_C.wandb_upload_client_list = [0, 1] # 0 is the server
_C.wandb_save_record_dataframe = False
_C.wandb_offline = False
_C.wandb_record = True


# ---------------------------------------------------------------------------- #
# mode settings
# ---------------------------------------------------------------------------- #
_C.mode = 'standalone'  # standalone or centralized
_C.test = True

# ---------------------------------------------------------------------------- #
# distributed settings
# ---------------------------------------------------------------------------- #
_C.client_num_in_total = 100
_C.client_num_per_round = 10
_C.instantiate_all = True
_C.clear_buffer = True
_C.aggregate_in_parallel = False

# ---------------------------------------------------------------------------- #
# device settings
# ---------------------------------------------------------------------------- #
_C.is_mobile = 0


# ---------------------------------------------------------------------------- #
# cluster settings
# ---------------------------------------------------------------------------- #
_C.rank = 0
_C.client_index = 0
_C.gpu_server_num = 1
_C.gpu_util_file = None
_C.gpu_util_key = None
_C.gpu_util_parse = None
_C.cluster_name = None

_C.gpu_index = 0  # for centralized training or standalone usage

# ---------------------------------------------------------------------------- #
# task settings
# ---------------------------------------------------------------------------- #
_C.task = 'classification' #    ["classification", "stackoverflow_lr", "ptb"]




# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.dataset = 'cifar10'
_C.dataset_aug = "default"
_C.dataset_resize = False
_C.dataset_load_image_size = 32
_C.num_classes = 10
_C.data_efficient_load = True    #  Efficiently load dataset, only load one full dataset, but split to many small ones.
_C.data_save_memory_mode = False    #  Clear data iterator, for saving memory, but may cause extra time.
_C.data_dir = './../../../data/cifar10'
_C.partition_method = 'iid'
_C.partition_alpha = 0.5
_C.dirichlet_min_p = None #  0.001    set dirichlet min value for letting each client has samples of each label
_C.dirichlet_balance = False # This will try to balance dataset partition among all clients to make them have similar data amount


_C.if_timm_dataset = False
_C.data_load_num_workers = 4

_C.an4_audio_path = " " # an4 audio paht
_C.lstm_num_steps = 35 # used for ptb, lstm_num_steps
_C.lstm_clip_grad = True
_C.lstm_clip_grad_thres = 0.25
_C.lstm_embedding_dim = 8
_C.lstm_hidden_size = 256



# ---------------------------------------------------------------------------- #
# data sampler
# ---------------------------------------------------------------------------- #
_C.data_sampler = "Random"



# ---------------------------------------------------------------------------- #
# data_preprocessing
# ---------------------------------------------------------------------------- #
_C.data_transform = "NormalTransform"  # or FLTransform
_C.TwoCropTransform = False


# ---------------------------------------------------------------------------- #
# checkpoint_save
# ---------------------------------------------------------------------------- #
_C.checkpoint_save = False
_C.checkpoint_save_model = False
_C.checkpoint_save_optim = False
_C.checkpoint_save_train_metric = False
_C.checkpoint_save_test_metric = False
_C.checkpoint_root_path = "./checkpoints/"
_C.checkpoint_epoch_list = [10, 20, 30]
_C.checkpoint_file_name_save_list = ["model", "dataset"]
_C.checkpoint_custom_name = "default"

_C.image_save_path = "./checkpoints/"

# ---------------------------------------------------------------------------- #
# record config
# ---------------------------------------------------------------------------- #
_C.record_dataframe = False
_C.record_level = 'epoch'   # iteration




# ---------------------------------------------------------------------------- #
# model_dif track
# ---------------------------------------------------------------------------- #
_C.model_dif_track = False
_C.model_dif_epoch_track = False
_C.model_dif_whole_track = False
_C.model_dif_LP_list = ['2']
_C.model_dif_divergence_track = False
_C.model_layer_dif_divergence_track = False
_C.model_rotation_epoch_track = False
_C.model_rotation_track = False
_C.model_layer_SVD_similarity_track = False
_C.model_layer_Cosine_similarity_track = False
_C.model_dif_client_list = [0, 1]
_C.model_dif_layers_list = ["layer1", "layer2"]

_C.model_dif_seq_FO_track = False
_C.model_dif_seq_SO_track = False
_C.model_dif_seq = [90]

_C.model_classifier_track = False

_C.model_layer_track = False   # means track layers
_C.model_whole_track = False   # means track whole model




# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.model = 'resnet20'
_C.model_input_channels = 3
_C.model_out_feature = False
_C.model_out_feature_layer = "last"
_C.model_feature_dim = 512
_C.model_output_dim = 10
_C.pretrained = False
_C.pretrained_dir = " "
_C.pretrained_submodel = False
_C.pretrained_layers = "Before-layer2"
_C.pretrained_model_name = None

_C.group_norm_num = 0


# ---------------------------------------------------------------------------- #
# Decouple Gradient Variance
# ---------------------------------------------------------------------------- #
_C.fed_split = None                
_C.fed_split_module_num = 2
_C.fed_split_module_choose = "layer2"
_C.fed_split_noise_mean = 0.5
_C.fed_split_noise_std = 0.1
_C.fed_split_client_mean_DP = None
_C.fed_split_client_DP_degree = 0.001
_C.fed_split_std_mode = "update"                # update or default, Defaul mode will use the ``fed_split_noise_std''
_C.fed_split_estimate_mode = "selected"              # ``all`` means using all clients, ``selected`` means current clients.
_C.fed_split_estimate_weight = "uniform"              # ``label_size`` means considering label distribution,  ``uniform`` means same.
_C.fed_split_loss_weight = 1.0
_C.fed_split_constrain_feat = True              # Add feature SupCon loss.   ``xxxConstrainxxx'' must use this
_C.fed_split_generator_loss = "MMD"             # MMD OT
_C.fed_split_feature_synthesis = "GaussianSynthesisLabel"
_C.fed_split_feature_weight_sched = "default"         # default, or gradual, or milestone, or gradual2half, gradualhalfhalf
_C.fed_split_feature_weight_max_epochs = 1000
_C.fed_split_hidden_loss = None            # SimCLR
_C.fed_split_hidden_loss_weight = 1.0
_C.fed_split_hidden_weight_sched = "default"
_C.fed_split_hidden_weight_max_epochs = 1000
_C.fed_split_CL_encoder_config = '1c2f'
_C.fed_split_CL_hidden_dim = 128
_C.fed_split_CL_b_temperature = 1.0          # Temperature
_C.fed_split_CL_temperature = 10.0          # Temperature
_C.fed_split_hidden_detach = False              # Whether detach hidden features.
_C.fed_split_forward_decouple = False           # Whether decouple forwarded features.
_C.fed_split_forward_detach_decouple_begin_epochs = 0           # detach or decouple sched epochs
_C.fed_split_freeze_front = False
_C.fed_split_freeze_begin_epochs = 500
_C.fed_split_decoder_config = '0f2c'            # The decoder model.



# ---------------------------------------------------------------------------- #
# Contrastive
# ---------------------------------------------------------------------------- #
_C.Contrastive = "no"                   # SimCLR, SupCon




# ---------------------------------------------------------------------------- #
# Client Select
# ---------------------------------------------------------------------------- #
_C.client_select = "random"  #   ood_score, ood_score_oracle



# ---------------------------------------------------------------------------- #
# Average weight
# ---------------------------------------------------------------------------- #
"""[even, datanum, inv_datanum, inv_datanum2datanum, even2datanum,
        ]
"""
# datanum2others is not considerred for now.
_C.fedavg_avg_weight_type = 'datanum'   




# ---------------------------------------------------------------------------- #
# Dif local steps
# ---------------------------------------------------------------------------- #
_C.fedavg_local_step_type = 'whole'   # whole, fixed, fixed2whole
_C.fedavg_local_step_fixed_type = 'lowest'   # default, lowest, highest, averaged
_C.fedavg_local_step_num = 10    # used for the fixed local step default 




# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'
""" ['CrossEntropy', 'nll_loss', 'LDAMLoss', 'local_LDAMLoss',
        'FocalLoss', 'local_FocalLoss']
"""
_C.normal_supcon_loss = False



# ---------------------------------------------------------------------------- #
# Imbalance weight
# ---------------------------------------------------------------------------- #
_C.imbalance_loss_reweight = False



# ---------------------------------------------------------------------------- #
# trainer
#---------------------------------------------------------------------------- #
# ['normal',  'lstm', 'nas']
_C.trainer_type = 'normal'


# ---------------------------------------------------------------------------- #
# algorithm settings
# ---------------------------------------------------------------------------- #
_C.algorithm = 'PSGD'
_C.psgd_exchange = 'grad' # 'grad', 'model'
_C.psgd_grad_sum = False
_C.psgd_grad_debug = False
_C.if_get_diff = False # this is suitable for other PS algorithms
_C.exchange_model = True

# Local SGD
_C.local_round_num = 4



# torch_ddp
_C.local_rank = 0
_C.init_method = 'tcp://127.0.0.1:23456'


# hvd settings and maybe used in future
_C.FP16 = False
_C.logging_gradients = False
_C.merge_threshold = 0
# horovod version
_C.hvd_origin = False
_C.nsteps_update = 1
_C.hvd_momentum_correction = 0 # Set it to 1 to turn on momentum_correction
_C.hvd_is_sparse = False






# fedprox
_C.fedprox = False
_C.fedprox_mu = 0.1

# fedavg
_C.fedavg_label_smooth = 0.0


# scaffold
_C.scaffold = False


# ---------------------------------------------------------------------------- #
# compression Including:
# 'topk','randomk', 'gtopk', 'randomkec',  'eftopk', 'gtopkef'
# 'quantize', 'qsgd', 'sign'
# ---------------------------------------------------------------------------- #
_C.compression = None

_C.compress_ratio = 1.0
_C.quantize_level = 32
_C.is_biased = 0

# ---------------------------------------------------------------------------- #
# optimizer settings
# comm_round is only used in FedAvg.
# ---------------------------------------------------------------------------- #
_C.max_epochs = 90
_C.global_epochs_per_round = 1
_C.comm_round = 90
_C.client_optimizer = 'no' # Please indicate which optimizer is used, if no, set it as 'no'
_C.server_optimizer = 'no'
_C.batch_size = 32
_C.lr = 0.1
_C.wd = 0.0001
_C.momentum = 0.9
_C.nesterov = False
_C.clip_grad = False



# ---------------------------------------------------------------------------- #
# Learning rate schedule parameters
# ---------------------------------------------------------------------------- #
_C.sched = 'no'   # no (no scheudler), StepLR MultiStepLR  CosineAnnealingLR
_C.lr_decay_rate = 0.992
_C.step_size = 1
_C.lr_milestones = [30, 60]
_C.lr_T_max = 10
_C.lr_eta_min = 0
_C.lr_warmup_type = 'constant' # constant, gradual.
_C.warmup_epochs = 0
_C.lr_warmup_value = 0.1




_C.freeze_backbone = False
_C.freeze_backbone_layers = "Before-layer2"
_C.freeze_bn = False




# ---------------------------------------------------------------------------- #
# Pruning
# ---------------------------------------------------------------------------- #
_C.pruning = False       #   enable or not pruning   
_C.gate_layer = False      # whether load models from gate_models    
_C.dynamic_network = None  # xxxxxxxxxxxxxxxxxxx
_C.fixed_network = False   # xxxxxxxxxxxxxxxxxxxxxxxxxxxx
_C.group_wd_coeff = 0.0  # ............
_C.flops_regularization = 0.0

_C.pruning_mask_from = ''
_C.compute_flops = False

_C.pruning_method = 'WEIGHT'    # xxxxxxxxxx
_C.pruning_momentum = 0.0
_C.pruning_step = 15
_C.prune_per_iteration = 10
_C.start_pruning_after_n_iterations = 0
_C.prune_neurons_max = -1
_C.maximum_pruning_iterations = 1e8

_C.pruning_silent = False
_C.l2_normalization_per_layer = False
_C.pruning_fixed_criteria = False
_C.pruning_starting_neuron = 0
_C.pruning_frequency = 30
_C.pruning_threshold = 100
_C.pruning_fixed_layer = 0
_C.pruning_combination_ID = 0
_C.pruning_group_size = 1
_C.maximum_pruning_iterations = 0
_C.maximum_pruning_iterations = 0











# ---------------------------------------------------------------------------- #
# Regularation
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# Evaluate settings
# ---------------------------------------------------------------------------- #
_C.frequency_of_the_test = 1



# ---------------------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------------------- #
_C.level = 'INFO' # 'INFO' or 'DEBUG'




# ---------------------------------------------------------------------------- #
# other settings
# ---------------------------------------------------------------------------- #
_C.ci = 0
_C.seed = 0






