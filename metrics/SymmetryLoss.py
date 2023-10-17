import torch
from torch import nn

from .loss_interface import LossInterface


class SymmetryLoss(LossInterface):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def evaluate(self, img, normalization=False):
        img = img.to(self.device)

        mseloss = nn.MSELoss()
        cur_loss = mseloss(img, torch.flip(img, [3]))

        # Loss must be multiplied by a negative value to obtain fitness
        symmetry_fitness = cur_loss / 10.0

        return symmetry_fitness

