import logging
import os
from re import L

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.utils.stats import record_net_data_stats


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLINETS_NUM = 3400
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'pixels'
_LABEL = 'label'


class TensorDataset(torch.utils.data.Dataset):
    
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)
        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_index=None):

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')
    train_x = []
    test_x = []
    train_y = []
    test_y = []

    # load data
    if client_index is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_index]]
        test_ids = [client_ids_test[client_index]]

    # load data in numpy format from h5 file
    train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
    train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
    test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
    test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()

    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    def expand_channel_dim(data):
        data = data.unsqueeze(0)
        return data
    train_transform.transforms.append(expand_channel_dim)
    test_transform.transforms.append(expand_channel_dim)

    # dataloader
    # train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long), transforms=train_transform)
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)

    # test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long), transforms=test_transform)
    test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_emnist(process_id, dataset, data_dir, 
                                                     batch_size=DEFAULT_BATCH_SIZE):

    if process_id == 0:
        # get global dataset
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = len(train_data_global)
        # logging.info("train_dl_global number = " + str(train_data_num))
        # logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            global client_ids_train, client_ids_test
            client_ids_train = list(train_h5[_EXAMPLE].keys())
            client_ids_test = list(test_h5[_EXAMPLE].keys())
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = local_data_num = len(train_data_local)
        train_data_global = None
        test_data_global = None

    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLINETS_NUM)]))
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_federated_emnist(dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):

    # client ids
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())

    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    traindata_cls_counts = {}
    for client_index in range(DEFAULT_TRAIN_CLINETS_NUM):
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_index)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_index] = local_data_num
        # logging.info("client_index = %d, local_sample_number = %d" % (client_index, local_data_num))
        # logging.info("client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_index, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_index] = train_data_local
        test_data_local_dict[client_index] = test_data_local
        unq, unq_cnt = np.unique(train_data_local.dataset.target_tensor.numpy(), return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        traindata_cls_counts[client_index] = tmp

    # global dataset
    train_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)
    
    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLINETS_NUM)]))
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts

