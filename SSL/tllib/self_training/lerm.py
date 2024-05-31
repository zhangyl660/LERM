
import torch
import torch.nn as nn
import torch.nn.functional as F


class LERM(nn.Module):
    """

    Inputs:
        - y_strong: unnormalized classifier predictions on strong augmented samples.
        - y: unnormalized classifier predictions on weak augmented samples.

    Shape:
        - y, y_strong: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    """

    def __init__(self,):
        super(LERM, self).__init__()


    def forward(self, y_strong, y):

        softmax_tgt = nn.Softmax(dim=1)(y_strong)
        a = torch.sum(softmax_tgt, dim=0)
        H = torch.mm(softmax_tgt.T, softmax_tgt)
        criterion = torch.nn.L1Loss()
        center_labels = torch.eye(H.size(dim=0))
        center_labels = center_labels.cuda()
        transfer_loss_1 = criterion((H.T / a).T, center_labels)

        softmax_tgt = nn.Softmax(dim=1)(y)
        a = torch.sum(softmax_tgt, dim=0)
        H = torch.mm(softmax_tgt.T, softmax_tgt)
        center_labels = torch.eye(H.size(dim=0))
        center_labels = center_labels.cuda()
        transfer_loss_2 = criterion((H.T / a).T, center_labels)
        
        transfer_loss = transfer_loss_1*0.1 + transfer_loss_2

        return transfer_loss
