from networks.invertible_generator import InvertibleGenerator
from networks.discriminator import Discriminator_patch
import torch
from torch import nn, optim

class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_id))

        self.model_names = ['netG', 'netD']

        self.netG = InvertibleGenerator(in_channel=opt.in_channel, n_block=opt.n_block, squeeze_num=opt.squeeze_num, conv_lu=opt.conv_lu, block_type=opt.block_type)
        self.netD = Discriminator_patch()

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr)

        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50000, gamma=0.5, last_epoch=-1)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50000, gamma=0.5, last_epoch=-1)

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

    def set_input(self, LDCT, SDCT):
        self.real_LDCT = LDCT.to(self.device)
        self.real_SDCT = SDCT.to(self.device)

    def set_input_val(self, LDCT, SDCT):
        self.real_LDCT = LDCT.to(self.device)
        self.real_SDCT = SDCT.to(self.device)

    def forward(self):
        self.fake_SDCT = self.netG(self.real_LDCT)

    def val_forward(self):
        self.fake_SDCT = self.netG(self.real_LDCT)
        self.cycle_LDCT = self.netG.inverse(self.fake_SDCT)
        self.fake_LDCT = self.netG.inverse(self.real_SDCT)

    def backward_G(self):
        fake_SDCT_logits = self.netD(self.fake_SDCT)
        self.Gan_loss = self.criterion_GAN(fake_SDCT_logits, torch.ones_like(fake_SDCT_logits))
        self.L1_loss = self.criterion_L1(self.real_LDCT, self.fake_SDCT)

        self.total_g_loss = self.opt.lambda_gan * self.Gan_loss + self.opt.lambda_l1 * self.L1_loss
        self.total_g_loss.backward()

    def backward_D(self):
        pred_real = self.netD(self.real_SDCT)
        D_real_loss = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake = self.netD(self.fake_SDCT.detach())
        D_fake_loss = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        self.total_D_loss = (D_real_loss + D_fake_loss)/2
        self.total_D_loss.backward()

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    # main loop
    def optimize_parameters(self):

        self.forward()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.scheduler_G.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.scheduler_D.step()

    def test(self):
        with torch.no_grad():
            self.forward()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()