import os
import time
import random
import datetime
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import data_loader
import numpy as np
import torch.nn as nn
import scipy.io as sio
from collections import defaultdict
from models import Prototypical, Discriminator
from loss import classification_loss_func, explicit_semantic_alignment_loss_func, knowledge_distillation_loss_func, \
    get_prototype_label
from utils import write_log_record, seed_everything, make_dirs, cost_matrix, structure_metrix_relation

from sklearn.svm import SVC
from keypointguide_POT.sinkhorn import sinkhorn_log_domain

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='Rethinking Guidance Information to Utilize Unlabeled Samples: A Label-Encoding Perspective')

parser.add_argument('--source', type=str, default='SAS', help='Source domain')
parser.add_argument('--target', type=str, default='TAD', help='Target domain')
parser.add_argument('--partition', type=int, default=20, help='Number of partition')
parser.add_argument('--reg', type=float, default=0.005,help="A parameter in OT")
parser.add_argument('--save_path', type=str, default='./KPG/', help='All records save path')
parser.add_argument('--seed', type=int, default=2020, help='seed for everything')

args = parser.parse_args()
args.time_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if len(args.cuda) == 1:
        torch.cuda.set_device(int(args.cuda))

# seed for everything
seed_everything(args)
# make dirs
# make_dirs(args)

if __name__ == '__main__':
    root_path = args.save_path
    if not os.path.exists(root_path):
        os.system('mkdir -p ' + root_path)
    result_best = 0.
    result = 0.
    acc_ssan_best = np.zeros((1, args.partition))
    acc_ssan = np.zeros((1, args.partition))

    # partition
    source_data_output = {"source_features": [], "source_labels": []}
    # partition
    target_data_output = {"training_features": [], "training_labels": [],
                   "testing_features": [], "testing_labels": []}

    for i in range(args.partition):
        configuration = data_loader.get_configuration(args, i)
        # configuration['source_data'] = [i.numpy() for i in configuration['source_data']]
        # configuration['labeled_target_data'] = [i.numpy() for i in configuration['labeled_target_data']]
        # configuration['unlabeled_target_data'] = [i.numpy() for i in configuration['unlabeled_target_data']]

        # numpy format
        source_data = [i.numpy() for i in configuration['source_data']]
        l_target_data = [i.numpy() for i in configuration['labeled_target_data']]
        u_target_data = [i.numpy() for i in configuration['unlabeled_target_data']]

        source_feature, source_label = source_data[0], source_data[1].reshape(-1, )
        l_target_feature, l_target_label = l_target_data[0], l_target_data[1].reshape(-1, )
        u_target_feature = u_target_data[0]

        ####key point
        I = []
        J = []
        t = 0
        feat_sl = []
        for l in l_target_label:
            I.append(t)
            J.append(t)
            fl = source_feature[source_label == l]
            feat_sl.append(np.mean(fl, axis=0))
            t += 1
        feat_sl = np.vstack(feat_sl)
        feat_s_ = np.vstack((feat_sl, source_feature))
        feat_t_ = np.vstack((l_target_feature, u_target_feature))
        Cs = cost_matrix(feat_s_, feat_s_)
        Cs /= Cs.max()
        Ct = cost_matrix(feat_t_, feat_t_)
        Ct /= Ct.max()
        p = np.ones(len(Cs)) / len(Cs)
        q = np.ones(len(Ct)) / len(Ct)
        C = structure_metrix_relation(Cs, Ct, I, J)
        C = C / C.max()
        ###mask
        M = np.ones_like(C)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1
        print("solving kpg-ot...")
        pi = sinkhorn_log_domain(p, q, C, M, reg=args.reg)
        feat_s_trans = pi @ feat_t_ / p.reshape(-1, 1)

        source_feature = feat_s_trans[len(l_target_feature):]

        configuration['source_data'] = [torch.from_numpy(source_feature), torch.from_numpy(source_label)]
        configuration['labeled_target_data'] = [torch.from_numpy(l_target_feature), torch.from_numpy(l_target_label)]
        configuration['unlabeled_target_data'] = [torch.from_numpy(u_target_feature),
                                                  torch.from_numpy(u_target_data[1])]
        source_data_output["source_features"].append(source_feature)
        source_data_output["source_labels"].append(source_label)
        target_data_output["training_features"].append(l_target_feature)
        target_data_output["training_labels"].append(l_target_label)
        target_data_output["testing_features"].append(u_target_feature)
        target_data_output["testing_labels"].append(u_target_data[1])

    source_path = os.path.join(root_path, f"Source_{args.source}2{args.target}.mat")
    target_path = os.path.join(root_path, f"Target_{args.target}.mat")

    for key in source_data_output:
        source_data_output[key] = np.array(source_data_output[key])
    for key in target_data_output:
        target_data_output[key] = np.array(target_data_output[key])

    sio.savemat(source_path, source_data_output)
    sio.savemat(target_path, target_data_output)
