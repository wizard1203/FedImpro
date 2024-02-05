
wandb_record=False wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &



wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &



wandb_record=False wandb_offline=True gpu_index=1 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &





wandb_record=True wandb_offline=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.1 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &







wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=128 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &



wandb_record=True wandb_offline=True gpu_index=1 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=128 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &





wandb_record=True wandb_offline=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=128 lr=0.1 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &




wandb_record=True wandb_offline=True gpu_index=3 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=128 lr=0.3 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
infopro_conf=infoproK2 fed_split_module_num=1 infopro_module_num=2 infopro_IXH=False infopro_IYH=False fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradualhalfhalf fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &










wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg  bash DecoupleGradVariance/run.sh &



wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg  bash DecoupleGradVariance/run.sh &




wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.1 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg  bash DecoupleGradVariance/run.sh &






wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   fedprox=True  bash DecoupleGradVariance/run.sh &




wandb_record=True wandb_offline=True gpu_index=1 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   fedprox=True  bash DecoupleGradVariance/run.sh &



wandb_record=True wandb_offline=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.1 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   fedprox=True  bash DecoupleGradVariance/run.sh &






wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.01 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   algorithm=FedNova  bash DecoupleGradVariance/run.sh &


wandb_record=True wandb_offline=True gpu_index=1 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   algorithm=FedNova  bash DecoupleGradVariance/run.sh &

wandb_record=True wandb_offline=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.5 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   algorithm=FedNova  bash DecoupleGradVariance/run.sh &








wandb_record=True wandb_offline=True gpu_index=0 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg  bash DecoupleGradVariance/run.sh &



wandb_record=True wandb_offline=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   fedprox=True  bash DecoupleGradVariance/run.sh &


wandb_record=True gpu_index=2 cluster_conf=localhost task_conf=femnist client_number_conf=client20 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
algo_conf=fedavg   algorithm=FedNova  bash DecoupleGradVariance/run.sh &














wandb_record=True wandb_offline=True gpu_index=1 cluster_conf=localhost task_conf=femnist client_number_conf=client50 model=resnet18_v2 \
client_num_in_total=3400 client_num_per_round=50 \
batch_size=20 lr=0.05 momentum=0.0 sched=no instantiate_all=False \
global_epochs_per_round=5 max_epochs=500 \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash DecoupleGradVariance/run.sh &
















