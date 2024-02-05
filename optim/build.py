import logging

import torch


from utils.data_utils import scan_model_with_depth
from utils.model_utils import build_param_groups

# Discard
# from .fedprox import FedProx


"""
    args.opt in 
    ["sgd", "adam"]
    --lr
    --momentum
    --clip-grad # wait to be developed
    --weight-decay, --wd
"""



def create_optimizer(args, model=None, params=None, **kwargs):
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
    else:
        if model is not None:
            params_to_optimizer = model.parameters()
        else:
            raise NotImplementedError
        pass

    if args.fed_split is not None or args.freeze_backbone:
        named_parameters, named_modules, layers_depth = scan_model_with_depth(model)
        param_groups = build_param_groups(named_parameters, named_modules, layers_depth)
    assert params_to_optimizer is not None

    if args.fed_split is not None or args.freeze_backbone:
        if args.server_optimizer == "sgd":
            optimizer = torch.optim.SGD(param_groups,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        elif args.server_optimizer == "adam":
            optimizer = torch.optim.Adam(param_groups,
                lr=args.lr, weight_decay=args.wd, amsgrad=True)
        elif args.server_optimizer == "no":
            optimizer = torch.optim.SGD(param_groups,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise NotImplementedError
    else:
        if (role == 'server') and (args.algorithm in [
            'FedAvg']):
            if args.server_optimizer == "sgd":
                # optimizer = torch.optim.SGD(params_to_optimizer,
                #     lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            elif args.server_optimizer == "adam":
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_optimizer),
                    lr=args.lr, weight_decay=args.wd, amsgrad=True)
            elif args.server_optimizer == "no":
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            else:
                raise NotImplementedError
        else:
            if args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(params_to_optimizer,
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            elif args.client_optimizer == "adam":
                raise NotImplementedError
            elif args.client_optimizer == "no":
                optimizer = torch.optim.SGD(params_to_optimizer,
                    lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            else:
                raise NotImplementedError

    return optimizer







