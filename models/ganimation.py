import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as numpy


class GANimation(BaseModel):
    def __init__(self, opt):
        super(GANimation, self).__init__(opt)
        self._name = 'GANimation'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G, device_ids=self._gpu_ids)
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D, device_ids=self._gpu_ids)
        self._D.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name('generator_wasserstein_gan', c_dim=self._opt.cond_nc)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_wasserstein_gan', c_dim=self._opt.cond_nc)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
            betas = [self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
            betas = [self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_real_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_real_cond = self._Tensor(self._opt.batch_size, self.cond_nc)
        self._input_desired_cond  = self._Tensor(self._opt.batch_size, self._opt.cond_nc)

    def _init_losses(self):
        # define loss functions
        self._criterion_cycle = torch.nn.L1Loss().cuda()
        self._criterion_D_cond = torch.nn.MSELoss().cuda()

        # init losses G
        self._loss_g_fake = Variable(self._Tensor([0]))
        self._loss_g_cond = Variable(self._Tensor([0]))
        self._loss_g_cyc = Variable(self._Tensor([0]))
        self._loss_g_mask_1 = Variable(self._Tensor([0]))
        self._loss_g_mask_2 = Variable(self._Tensor([0]))
        self._loss_g_idt = Variable(self._Tensor([0]))
        self._loss_g_masked_fake = Variable(self._Tensor([0]))
        self._loss_g_masked_cond = Variable(self._Tensor([0]))
        self._loss_g_mask_1_smooth = Variable(self._Tensor([0]))
        self._loss_g_mask_2_smooth = Variable(self._Tensor([0]))
        self._loss_rec_real_img_rgb = Variable(self._Tensor([0]))
        self._loss_g_fake_imgs_smooth = Variable(self._Tensor([0]))
        self._loss_g_unmasked_rgb = Variable(self._Tensor([0]))

        # init losses D
        self._loss_d_real = Variable(self._Tensor([0]))
        self._loss_d_cond = Variable(self._Tensor([0]))
        self._loss_d_fake = Variable(self._Tensor([0]))
        self._loss_d_gp = Variable(self._Tensor([0]))

    def set_input(self, input):
        self._input_real_img.resize_(input['real_img'].size()).copy_(input['real_img'])
        self._input_real_cond.resize_(input['real_cond'].size()).copy_(input['real_cond'])
        self._input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
        self._input_real_id = input['sample_id']
        self._input_real_img_path = input['real_img_path']

        if len(self._gpu_ids) > 0:
            self._input_real_im = self._input_real_img.cuda(self._gpu_ids[0], async=True)
            self._input_real_cond = self._input_real_cond.cuda(self._gpu_ids[0], async=True)
            self._input_desired_cond = self._input_desired_cond.cuda(self._gpu_ids[0], async=True)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    