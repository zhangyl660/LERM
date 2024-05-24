import argparse
import os
import os.path as osp

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import network
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
       iter_test = iter(loader["test"])
       for i in range(len(loader['test'])):
           data = next(iter_test)
           inputs = data[0]
           labels = data[1]
           inputs = inputs.cuda()
           labels = labels.cuda()
           _, outputs = model(inputs)
           if start_test:
               all_output = outputs.float()
               all_label = labels.float()
               start_test = False
           else:
               all_output = torch.cat((all_output, outputs.float()), 0)
               all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]
    if "webcam" in data_config["source"]["list_path"] or "dslr" in data_config["source"]["list_path"]:
        prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    else:
        prep_dict["source"] = prep.image_target(**config["prep"]['params'])  # TODO

    if "webcam" in data_config["target"]["list_path"] or "dslr" in data_config["target"]["list_path"]:
        prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    else:
        prep_dict["target"] = prep.image_target(**config["prep"]['params'])  # TODO

    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network.load_state_dict(model_state_dict)

    base_network = base_network.cuda()

    Acc = image_classification_test(dset_loaders, base_network)
    print(Acc)


    # multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i, k in enumerate(gpus)])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    print('len_train_source', len_train_source)
    print('len_train_target', len_train_target)
    best_acc = 0.0


    base_network.train(False)
    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])

    inputs_source, labels_source = next(iter_source)
    inputs_source = inputs_source.cuda()
    features_source, outputs_source = base_network(inputs_source)
    Pro_Xs = features_source.cpu().detach().numpy()
    Ys = labels_source.numpy()

    inputs_target, labels_target = next(iter_target)
    inputs_target = inputs_target.cuda()
    features_target, outputs_target = base_network(inputs_target)
    Pro_Xu = features_target.cpu().detach().numpy()
    Yu = labels_target.numpy()

    for inputs_source, labels_source in iter_source:
        inputs_source = inputs_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        Pro_Xs = np.concatenate((Pro_Xs, features_source.cpu().detach().numpy()), axis=0)
        Ys = np.concatenate((Ys, labels_source.detach().numpy()), axis=0)

    for inputs_target, labels_target in iter_target:
        inputs_target = inputs_target.cuda()
        features_target, outputs_target = base_network(inputs_target)
        Pro_Xu = np.concatenate((Pro_Xu, features_target.cpu().detach().numpy()), axis=0)
        Yu = np.concatenate((Yu, labels_target.numpy()), axis=0)

    matdata_output = {"Pro_Xs": Pro_Xs, "Pro_Xu": Pro_Xu, "Ys": Ys, "Yu": Yu}
    print(matdata_output)
    scipy.io.savemat(osp.join(config["output_path"], "features_labels.mat"), matdata_output)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet50")
    parser.add_argument('--dset', type=str, default='office-home', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office-home/Art.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office-home/Clipart.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--print_num', type=int, default=100, help="interval of two print loss")
    parser.add_argument('--num_iterations', type=int, default=6002, help="interation num ")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--method', type=str, default='BNM', help="Options: BNM, ENT, BFM, FBNM, NO")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for transfer loss")
    parser.add_argument('--batch_size', type=int, default=36, help="batch size")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["method"] = args.method
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.dset + "/" + args.output_dir

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": args.trade_off}
    if "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": False, "bottleneck_dim": 256,
                                        "new_cls": True}}
    else:
        raise ValueError('Network cannot be recognized. Please define your own dataset here.')

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": args.batch_size}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": args.batch_size}}

    if config["dataset"] == "office":
        if ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    print('begin')
    best_acc = train(config)
    config["out_file"].write(str(best_acc))
    config["out_file"].flush()
    print("best", best_acc)
