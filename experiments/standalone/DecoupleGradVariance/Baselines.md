


wandb_record=False level=DEBUG \
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  fedprox=True   bash DecoupleGradVariance/run.sh &

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg scaffold=True   bash DecoupleGradVariance/run.sh &

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova  bash DecoupleGradVariance/run.sh &


wandb_record=False level=DEBUG \
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=femnist model=resnet18_v2 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &




























