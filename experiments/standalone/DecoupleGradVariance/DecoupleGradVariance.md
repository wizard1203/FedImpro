

## FeatureSynthesisLabel


wandb_record=False level=DEBUG \
gpu_index=2 cluster_conf=localhost \
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=resnet18_v2 algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh


wandb_record=False level=DEBUG \
gpu_index=2 cluster_conf=localhost \
checkpoint_save=True checkpoint_save_model=True checkpoint_root_path="./DecoupleGradVariance/checkpoints/"\
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=resnet18_v2 algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh



wandb_record=False level=DEBUG \

gpu_index=0 cluster_conf=localhost \
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=vgg11 algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=1000 \
bash DecoupleGradVariance/run.sh


gpu_index=3 cluster_conf=localhost \
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=vgg11 algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.01 \
fed_split_module_choose=layer2 \
fed_split_std_mode=default \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh













wandb_record=False level=DEBUG \
gpu_index=3 cluster_conf=localhost \
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=vgg11 fedprox=True algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh















