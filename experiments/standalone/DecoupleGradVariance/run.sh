#!/bin/bash

algo_conf=${algo_conf:-normal}
client_number_conf=${client_number_conf:-client10}
cluster_conf=${cluster_conf:-localhost}
task_conf=${task_conf:-cifar10}

echo $algo_conf


export entity="hpml-hkbu"
export project="friend_avg"
export level=${level:-INFO}
export exp_mode=${exp_mode:-"ready"}

source DecoupleGradVariance/client_number/${client_number_conf}.sh
source DecoupleGradVariance/tasks/${task_conf}.sh
source DecoupleGradVariance/algorithms/${algo_conf}.sh


export cluster_name=${cluster_name:-localhost}

# export WANDB_MODE=offline
export WANDB_MODE=online
export gpu_index=${gpu_index:-0}



export sched=${sched:-no}
# export lr_decay_rate=0.97
export lr_decay_rate=${lr_decay_rate:-0.992}
export momentum=${momentum:-0.9}
export global_epochs_per_round=${global_epochs_per_round:-1}
export max_epochs=${max_epochs:-1000}


export client_num_in_total=${client_num_in_total:-10}
export client_num_per_round=${client_num_per_round:-5}

# export dataset="cifar10"
# export model="vgg-9"

export dataset=${dataset:-cifar10}
export model=${model:-resnet18_v2}
export lr=${lr:-0.01}

export partition_method=${partition_method:-'hetero'}
export partition_alpha=${partition_alpha:-0.1}


export batch_size=${batch_size:-128}
export wd=${wd:-0.0001}

export record_dataframe=False
export wandb_save_record_dataframe=False
export wandb_upload_client_list="[]"



export checkpoint_save=${checkpoint_save:-False}
export checkpoint_save_model=${checkpoint_save_model:-False}
export checkpoint_file_name_save_list=${checkpoint_file_name_save_list:-"['mode','algorithm','model','dataset','batch_size','lr','sched',\
'partition_method','partition_alpha','pretrained'\
]"}
export checkpoint_epoch_list="[1,99,999]"
export corr_layers_list=""

export losses_track=${losses_track:-False}

export grad_track_layers_list="[\
'conv1','conv2',\
'fc1','fc2','fc3']"
export grad_track=False
export grad_norm_track=False
export grad_sum_norm_track=False
export grad_LP_list="['1','2','inf']"

export client_optimizer=${client_optimizer:-sgd}

export server_optimizer=${server_optimizer:-sgd}

export algorithm=${algorithm:-"FedAvg"}


export data_sampler=${data_sampler:-Random}

# export script=${script:-'./launch_standalone.sh'}

# export fedprox_mu=${fedprox_mu:-1.0}
export fedprox_mu=${fedprox_mu:-0.1}


# export script='./launch_standalone.sh'
bash launch_standalone.sh








