# DecoupleGradVariance

To launch the experiments, you can 

``` 
cd experiments/standalone
```


Note that you may need to config your data_dir and python path by yourself.

Then you can use the following commands to run baselines 
```
wandb_record=False level=DEBUG \
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash close_look_a/run.sh

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  fedprox=True   bash close_look_a/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg scaffold=True   bash close_look_a/run.sh

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova  bash close_look_a/run.sh
```

You can follow the following commands to run our algorithm. 
```
gpu_index=0 cluster_conf=localhost \
partition_alpha=0.1 task_conf=cifar10 client_number_conf=client10 model=resnet18_v2 algo_conf=fedavg \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
fed_split_loss_weight=1.0 \
algo_conf=fed_split_FeatureSynthesisLabel  \
fed_split_module_num=1 fed_split_constrain_feat=False \
fed_split_feature_synthesis=GaussianSynthesisLabel fed_split_noise_std=0.1 \
fed_split_module_choose=layer2 \
fed_split_std_mode=update \
fed_split_feature_weight_sched=gradual2half fed_split_feature_weight_max_epochs=500 \
bash close_look_a/run.sh
```






