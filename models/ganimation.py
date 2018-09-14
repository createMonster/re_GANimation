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

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    # get image paths
    def get_image_paths(self):
        return OrderedDict(['read_img', self._input_real_img_path])

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        if not self._is_train:
            # convert tensor to variables
            real_img = Variable(self._input_real_img, volatile=True)
            real_cond = Variable(self._input_real_cond, volatile=True)
            desired_cond = Variable(self._input_desired_cond, volatile=True)

            # generate fake images
            fake_imgs, fake_img_mask = self._G.forward(real_img, desired_cond)
            fake_img_mask = self.do_if_necessary_saturate_mask(rec_real_img_mask, saturate=self._opt.do_saturate_mask)
            fake_imgs_masked = fake_img_mask * real_img + (1 - fake_img_mask) * fake_imgs

            rec_real_img_rgb, rec_real_img_mask = self._G.forward(fake_imgs_masked, real_cond)
            rec_real_img_mask = self._do_if_necessary_saturate_mask(rec_real_img_mask, saturate=self._opt.do_saturate_mask)
            rec_real_imgs = rec_real_img_mask * fake_imgs_masked + (1 - rec_real_img_mask) * rec_real_img_rgb

            imgs = None
            data = None
            if return_estimates:
                # normalize mask for better visualization
                fake_img_mask_max = fake_imgs_masked.view(fake_img_mask.size(0), -1).max(-1)[0]
                fake_img_mask_max = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(fake_img_mask_max, -1), -1), -1)
                # fake_img_mask_norm = fake_img_mask / fake_img_mask_max
                fake_img_mask_norm = fake_img_mask

                # generate images
                im_real_img = util.tensor2im(real_img.data)
                im_fake_imgs = util.tensor2im(fake_imgs.data)
                im_fake_img_mask_norm = util.tensor2maskim(fake_img_mask_norm.data)
                im_fake_imgs_masked = util.tensor2im(fake_imgs_masked.data)
                im_rec_imgs = util.tensor2im(rec_real_img_rgb.data)
                im_rec_img_mask_norm = util.tensor2maskim(rec_real_img_mask.data)
                im_rec_imgs_masked = util.tensor2im(rec_real_imgs.data)
                im_concat_img = np.concatenate([im_real_img, im_fake_imgs_masked, im_fake_img_mask_norm, im_fake_imgs,
                                                im_rec_imgs, im_rec_img_mask_norm, im_rec_imgs_masked],
                                               1)

                im_real_img_batch = util.tensor2im(real_img.data, idx=-1, nrows=1)
                im_fake_imgs_batch = util.tensor2im(fake_imgs.data, idx=-1, nrows=1)
                im_fake_img_mask_norm_batch = util.tensor2maskim(fake_img_mask_norm.data, idx=-1, nrows=1)
                im_fake_imgs_masked_batch = util.tensor2im(fake_imgs_masked.data, idx=-1, nrows=1)
                im_concat_img_batch = np.concatenate([im_real_img_batch, im_fake_imgs_masked_batch,
                                                      im_fake_img_mask_norm_batch, im_fake_imgs_batch],
                                                     1)

                imgs = OrderedDict([('real_img', im_real_img),
                                    ('fake_imgs', im_fake_imgs),
                                    ('fake_img_mask', im_fake_img_mask_norm),
                                    ('fake_imgs_masked', im_fake_imgs_masked),
                                    ('concat', im_concat_img),
                                    ('real_img_batch', im_real_img_batch),
                                    ('fake_imgs_batch', im_fake_imgs_batch),
                                    ('fake_img_mask_batch', im_fake_img_mask_norm_batch),
                                    ('fake_imgs_masked_batch', im_fake_imgs_masked_batch),
                                    ('concat_batch', im_concat_img_batch),
                                    ])

                data = OrderedDict([('real_path', self._input_real_img_path),
                                    ('desired_cond', desired_cond.data[0, ...].cpu().numpy().astype('str'))
                                    ])

            # keep data for visualization
            if keep_data_for_visuals:
                self._vis_real_img = util.tensor2im(self._input_real_img)
                self._vis_fake_img_unmasked = util.tensor2im(fake_imgs.data)
                self._vis_fake_img = util.tensor2im(fake_imgs_masked.data)
                self._vis_fake_img_mask = util.tensor2maskim(fake_img_mask.data)
                self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
                self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()
                self._vis_batch_real_img = util.tensor2im(self._input_real_img, idx=-1)
                self._vis_batch_fake_img_mask = util.tensor2maskim(fake_img_mask.data, idx=-1)
                self._vis_batch_fake_img = util.tensor2im(fake_imgs_masked.data, idx=-1)

            return imgs, data

    def optimize_parameters(self, train_generator-True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_real_img.size(0)
            self._real_img = Variable(self._input_real_img)
            self._real_cond = Variable(self._input_real_cond)
            self._desired_cond = Variable(self._input_desired_cond)

            # train D
            loss_D, fake_imgs_masked = self._forward_D()
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

            loss_D_gp = self._gradient_penalty_D(fake_imgs_masked)
            self._optimizer_D.zero_grad()
            loss_D_gp.backward()
            self._optimizer_D.step()

            # train G
            if train_generator:
                loss_G = self._forward_G(keep_data_for_visuals)
                self._optimizer_G.zero_grad()
                loss_G.backward()
                self._optimizer_G.step()

    

    