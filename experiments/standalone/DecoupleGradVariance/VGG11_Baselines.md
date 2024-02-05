


wandb_record=False level=DEBUG \


gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  fedprox=True   bash DecoupleGradVariance/run.sh &

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg scaffold=True   bash DecoupleGradVariance/run.sh &

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova  bash DecoupleGradVariance/run.sh &




gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.03 momentum=0.9  sched=no \
client_number_conf=client10 algo_conf=fedavg  fedprox=True   bash DecoupleGradVariance/run.sh &

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.03 momentum=0.9  sched=no \
client_number_conf=client10 algo_conf=fedavg scaffold=True   bash DecoupleGradVariance/run.sh &

gpu_index=0 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.03 momentum=0.9  sched=no \
client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova  bash DecoupleGradVariance/run.sh &







gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.01 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &

gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.03 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &



gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.1 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 model=vgg11 \
batch_size=128 lr=0.003 momentum=0.0  sched=no \
client_number_conf=client10 algo_conf=fedavg   bash DecoupleGradVariance/run.sh &























