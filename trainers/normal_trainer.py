import copy
import logging
import time

import torch
import wandb
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.distributions import Categorical

from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from fedml_core.trainer.model_trainer import ModelTrainer

from data_preprocessing.utils.stats import record_batch_data_stats

from utils.data_utils import (
    get_data,
    get_named_data,
    get_all_bn_params,
    apply_gradient,
    clear_grad,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    check_device,
    get_train_batch_data
)

from utils.model_utils import (
    set_freeze_by_names,
    get_actual_layer_names,
    freeze_by_names,
    unfreeze_by_names,
    get_modules_by_names
)

from utils.context import (
    raise_error_without_process,
    get_lock,
)

from utils.checkpoint import (
    setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint,
    setup_checkpoint_file_name_prefix,
    save_checkpoint_without_check
)


from model.build import create_model

from trainers.averager import Averager
from trainers.spliter import Spliter


class NormalTrainer(ModelTrainer):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        super().__init__(model)

        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model
        # self.model.to(device)
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        self.save_checkpoints_config = setup_checkpoint_config(self.args)

        # For future use
        self.param_groups = self.optimizer.param_groups
        with raise_error_without_process():
            self.param_names = list(
                enumerate([group["name"] for group in self.param_groups])
            )

        self.named_parameters = list(self.model.named_parameters())

        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model)

        self.lr_scheduler = lr_scheduler
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step(0)
        if self.args.fed_split:
            self.spliter = Spliter(args, device=self.device, model=self.model, optimizer=self.optimizer)



    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        pass

    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def update_state(self, **kwargs):
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)
        self.update_optimizer_state(**kwargs)

    def update_loss_state(self, **kwargs):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss"]:
            kwargs['cls_num_list'] = kwargs["selected_cls_num_list"]
            self.criterion.update(**kwargs)
        elif self.args.loss_fn in ["local_FocalLoss", "local_LDAMLoss"]:
            kwargs['cls_num_list'] = kwargs["local_cls_num_list_dict"][self.index]
            self.criterion.update(**kwargs)

    def update_optimizer_state(self, **kwargs):
        pass


    def generate_fake_data(self, num_of_samples=64):
        input = torch.randn(num_of_samples, self.args.model_input_channels,
                    self.args.dataset_load_image_size, self.args.dataset_load_image_size)
        return input


    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)


    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_grads

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device))


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)

    # Used for single machine training
    # Should be discarded #TODO
    def train(self, train_data, device, args, **kwargs):
        model = self.model

        model.train()

        epoch_loss = []
        for epoch in range(args.max_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()

                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Train Epo: {} \tLoss: {:.6f}'.format(
                    self.index, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1)


    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def summarize(self, model, output, labels,
        tracker, metrics,
        loss,
        epoch, batch_idx,
        mode='train',
        checkpoint_extra_name="centralized",
        things_to_track=[],
        if_update_timer=False,
        train_data=None, train_batch_data=None,
        end_of_epoch=None,
    ):
        # if np.isnan(loss.item()):
        # logging
        if np.isnan(loss):
            logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                iteration: {}, loss is nan!!!! '.format(
                self.index, epoch, batch_idx))
            # loss.data.fill_(100)
            loss = 100
        metric_stat = metrics.evaluate(loss, output, labels)
        tracker.update_metrics(
            metric_stat, 
            metrics_n_samples=labels.size(0)
        )

        if len(things_to_track) > 0:
            if end_of_epoch is not None:
                pass
            else:
                end_of_epoch = (batch_idx == len(train_data) - 1)
            self.track(tracker, self.args.batch_size, model, loss, end_of_epoch,
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track)

        if if_update_timer:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
                But only for epoch training, because One-step training will be scheduled by client or server
            """
            tracker.timer.past_iterations(iterations=1)

        if mode == 'train':
            logging.info('Trainer {}. Glob comm round: {}, Train Epo: {}, iter: {} '.format(
                self.index, tracker.timer.global_comm_round_idx, epoch, batch_idx) + metrics.str_fn(metric_stat))
                # logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
                #     self.index, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc1']))
        elif mode == 'test':
            logging.info('(Trainer_ID {}. Test epoch: {}, iteration: {} '.format(
                self.index, epoch, batch_idx) + metrics.str_fn(metric_stat))
        else:
            raise NotImplementedError
        return metric_stat



    def train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model

        if move_to_gpu:
            model.to(device)
        model.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fed_split:
            self.spliter.freeze_layers(progress=kwargs["global_outer_epoch_idx"])

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            if self.args.fed_split:
                output, feat_list, fake_labels_list, split_loss_value, split_hidden_loss, decode_error, loss = self.spliter.split_train(
                    model, x, labels, loss_func=self.criterion, progress=kwargs["global_outer_epoch_idx"])
                tracker.update_local_record(
                        'generator_track',
                        server_index=self.server_index,
                        client_index=self.client_index,
                        summary_n_samples=real_batch_size*1,
                        args=self.args,
                        split_loss_value=split_loss_value,
                        split_hidden_loss=split_hidden_loss,
                        split_decode_error=decode_error,
                    )
                if self.args.fedprox:
                    fed_prox_reg = 0.0
                    previous_model = kwargs["previous_model"]
                    for name, param in model.named_parameters():
                        fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                            torch.norm((param - previous_model[name].data.to(device)))**2)
                    prox_loss = fed_prox_reg
                    prox_loss.backward()
            else:
                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)

                loss = self.criterion(output, labels)
                if self.args.fedprox:
                    fed_prox_reg = 0.0
                    previous_model = kwargs["previous_model"]
                    for name, param in model.named_parameters():
                        fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                            torch.norm((param - previous_model[name].data.to(device)))**2)
                    loss += fed_prox_reg

            if self.args.fed_split:
                loss_value = loss
            else:
                # logging.info(f"output.shape: {output.shape}, labels.shape:{labels.shape}, labels:{labels}")
                loss.backward()
                loss_value = loss.item()
            self.optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                if self.args.fed_split and self.args.fed_split_forward_decouple:
                    # self.summarize(model, output, fake_labels_list[-1],
                    self.summarize(model, output, fake_labels_list[-1],
                            tracker, metrics,
                            loss_value,
                            epoch, batch_idx,
                            mode='train',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=True if self.args.record_dataframe else False,
                            train_data=train_data, train_batch_data=train_batch_data,
                            end_of_epoch=None,
                        )
                else:
                    self.summarize(model, output, labels,
                            tracker, metrics,
                            loss_value,
                            epoch, batch_idx,
                            mode='train',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=True if self.args.record_dataframe else False,
                            train_data=train_data, train_batch_data=train_batch_data,
                            end_of_epoch=None,
                        )


    def train_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):

        model = self.model

        if move_to_gpu:
            model.to(device)

        model.train()

        x, labels = train_batch_data

        if self.args.TwoCropTransform:
            x = torch.cat([x[0], x[1]], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        x, labels = x.to(device), labels.to(device)
        real_batch_size = labels.shape[0]

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        if self.args.fed_split:
            output, feat_list, fake_labels_list, split_loss_value, split_hidden_loss, decode_error, loss = self.spliter.split_train(
                model, x, labels, loss_func=self.criterion, progress=epoch)
            tracker.update_local_record(
                    'generator_track',
                    server_index=self.server_index,
                    client_index=self.client_index,
                    summary_n_samples=real_batch_size*1,
                    args=self.args,
                    split_loss_value=split_loss_value,
                    split_hidden_loss=split_hidden_loss,
                    split_decode_error=decode_error,
                )
        else:
            if self.args.model_out_feature:
                output, feat = model(x)
            else:
                output = model(x)
            # logging.info(f"output.shape: {output.shape},, labels.shape:{labels.shape}, labels:{labels}")
            loss = self.criterion(output, labels)

        if self.args.fed_split:
            pass
            loss_value = loss
        else:
            loss.backward()
            loss_value = loss.item()

        self.optimizer.step()

        if make_summary and (tracker is not None) and (metrics is not None):
            if self.args.fed_split and self.args.fed_split_forward_decouple:
                self.summarize(model, output, fake_labels_list[-1],
                        tracker, metrics,
                        loss_value,
                        epoch, iteration,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=False,
                        train_data=None, train_batch_data=train_batch_data,
                        end_of_epoch=end_of_epoch,
                    )
            else:
                # logging.info(f"")
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, iteration,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=False,
                        train_data=None, train_batch_data=train_batch_data,
                        end_of_epoch=end_of_epoch,
                    )

        return loss, output, labels


    def infer_bw_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, model_train=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):
        """
            inference and BP without optimization
        """
        model = self.model

        if move_to_gpu:
            model.to(device)

        if model_train:
            model.train()
        else:
            model.eval()

        time_table = {}
        time_now = time.time()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.model_out_feature:
            output, feat = model(x)
        else:
            output = model(x)
        loss = self.criterion(output, labels)
        loss_value = loss.item()
        time_table["FP"] = time.time() - time_now
        time_now = time.time()
        logging.debug(f" Whole model time FP: {time.time() - time_now}")

        loss.backward()

        if make_summary and (tracker is not None) and (metrics is not None):
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss_value,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels



    def test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):

        model = self.model
        Acc_accm = 0.0

        model.eval()
        if move_to_gpu:
            model.to(device)
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                real_batch_size = labels.shape[0]
                if self.args.model_input_channels ==3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)

                loss = self.criterion(output, labels)

                if make_summary and (tracker is not None) and (metrics is not None):
                    metric_stat = self.summarize(model, output, labels,
                            tracker, metrics,
                            loss.item(),
                            epoch, batch_idx,
                            mode='test',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=False,
                            train_data=test_data, train_batch_data=None,
                            end_of_epoch=False,
                        )
                    logging.debug(f"metric_stat[Acc1] is {metric_stat['Acc1']} ")
                    Acc_accm += metric_stat["Acc1"]
            logging.debug(f"Total is {Acc_accm} , averaged is {Acc_accm / (batch_idx+1)}")


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None,
                        epoch=None, iteration=None, tracker=None, metrics=None):
        pass









