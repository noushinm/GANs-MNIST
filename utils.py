import torch
import torch.nn as nn

def noise(bs, dim, is_cuda=False):
    """Generate random Gaussian noise vectors N(0,I)"""
    out = torch.randn((bs, dim))
    if is_cuda:
        out = out.cuda()
    return out

def DLoss(logits_real, logits_fake, targets_real, targets_fake):
    bce_loss = nn.BCEWithLogitsLoss()
    logits = torch.cat((logits_real, logits_fake))
    targets = torch.cat((targets_real, targets_fake))
    loss = bce_loss(logits, targets)
    return loss

def GLoss(logits_fake, targets_real):
    bce_loss = nn.BCEWithLogitsLoss()
    g_loss = bce_loss(logits_fake, targets_real)
    return g_loss