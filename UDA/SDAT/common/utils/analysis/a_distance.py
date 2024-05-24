"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD

from .. import ForeverDataIterator
from ..meter import AverageMeter
from ..metric import binary_accuracy


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=50):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    # feature = torch.cat([source_feature, target_feature], dim=0)
    # label = torch.cat([source_label, target_label], dim=0)

    source_dataset = TensorDataset(source_feature, source_label)
    target_dataset = TensorDataset(target_feature, target_label)
    print("source_dataset_len", len(source_dataset))
    print("target_dataset_len", len(target_dataset))

    # dataset = TensorDataset(feature, label)
    source_size = len(source_dataset)
    target_size = len(target_dataset)
    # train_size = int(0.8 * length)
    # val_size = length - train_size
    train_source_dataset, val_source_dataset = torch.utils.data.random_split(source_dataset, [int(0.8 * source_size), source_size-int(0.8 * source_size)])
    train_target_dataset, val_target_dataset = torch.utils.data.random_split(target_dataset, [int(0.8 * target_size), target_size-int(0.8 * target_size)])
    val_set = ConcatDataset([val_target_dataset, val_source_dataset])
    print("val len", len(val_set))
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

    # _, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_source_loader = DataLoader(train_source_dataset, batch_size=16, shuffle=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=16, shuffle=True)
    train_target_iter = ForeverDataIterator(train_target_loader)

    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(source_feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.005)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_source_loader:
            x_target, label_target = next(train_target_iter)
            label = torch.cat([label, label_target], dim=0)
            x = torch.cat([x, x_target], dim=0)
            x = x.to(device)
            label = label.to(device)
            # print(label.size(), x.size())
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance

