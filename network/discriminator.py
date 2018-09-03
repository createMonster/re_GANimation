import torch.nn as nn
import numpy as numpy
from .networks import NetworkBase

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN. """
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_wgan'

        