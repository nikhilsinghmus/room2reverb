import os
import torch
from .networks import Encoder, Generator, Discriminator
from .stft import STFT
from .util import estimate_t60


# Hyperparameters
G_LR = 2e-4
D_LR = 4e-4
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA = 100


class Room2Reverb:
    def __init__(self, encoder_path, latent_dimension=512):
        """GAN model class, puts everything together."""
        self._encoder_path = encoder_path
        self._latent_dimension = latent_dimension
        self._enc_dimension = self._latent_dimension if self._latent_dimension == 2048 else 365
        self._init_network()
        self._init_optimizer()
        self._criterion_GAN = torch.nn.MSELoss().cuda()
        self._criterion_extra = torch.nn.L1Loss().cuda()
        self.stft = STFT()

    def _init_network(self): # Initialize networks
        self.enc = Encoder(self._encoder_path, self._enc_dimension)
        self.g = Generator()
        self.d = Discriminator()
        self.g.cuda()
        self.d.cuda()
        
    def _init_optimizer(self): # Initialize optimizers
        self.g_optim = torch.optim.Adam(self.g.model.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.d_optim = torch.optim.Adam(self.d.model.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.enc_optim = torch.optim.Adam(self.enc.model.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)

    def train_step(self, spec, label, train_g, train_enc=True):
        """Perform one training step."""
        spec.requires_grad = True # For the backward pass, seems necessary for now
        
        # Forward passes through models
        f = self.enc.forward(label).cuda().detach()
        z = torch.randn(f.shape).cuda() if self._enc_dimension == 2048 else torch.randn((f.shape[0], self._latent_dimension - f.shape[1], f.shape[2], f.shape[3])).cuda()
        fake_spec = self.g(torch.cat((f, z), 1))
        d_fake = self.d(fake_spec.detach(), f)
        d_real = self.d(spec, f)

        # Wasserstein objective
        # mean_fake = d_fake.mean()
        # mean_real = d_real.mean()

        # Update the discriminator weights
        # self.d.zero_grad()
        # gradient_penalty = 10 * self.wgan_gp(spec.data, fake_spec.data)
        # self.D_loss = -mean_real + mean_fake.mean() + gradient_penalty
        # self.D_loss.backward()
        # self.d_optim.step()

        # self.g.zero_grad()
        # if train_g: # Train generator once every k iterations
            # d_fake = self.d(fake_spec)
            # self.G_loss = -mean_fake
            # self.G_loss.backward()
            # self.g_optim.step()

        # MSE + L1 version
        self.g.zero_grad()
        d_fake2 = self.d(fake_spec.detach(), f)
        # d_real2 = self.d(spec)
        G_loss1 = self._criterion_GAN(d_fake2, torch.ones(d_fake2.shape).cuda())
        G_loss2 = self._criterion_extra(fake_spec, spec)
        self.G_loss = G_loss1 + (LAMBDA * G_loss2)
        self.G_loss.backward()
        self.g_optim.step()

        if train_enc:
            self.enc_optim.step()

        self.d.zero_grad()
        l_fakeD = self._criterion_GAN(d_fake, torch.zeros(d_fake.shape).cuda())
        l_realD = self._criterion_GAN(d_real, torch.ones(d_real.shape).cuda())
        self.D_loss = (l_realD + l_fakeD)
        
        if self.D_loss > 0.2: # Avoid overly strong discriminator
            self.D_loss.backward()
            self.d_optim.step()
        
    def wgan_gp(self, real_data, fake_data): # Gradient penalty to promote Lipschitz continuity
        alpha = torch.rand(1, 1)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates = self.d(interpolates)
        fake = torch.Tensor(real_data.shape[0], 1).fill_(1.0).requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def load_generator(self, path): # Load a pre-trained generator
        self.g.load_state_dict(path)
    
    def load_discriminator(self, path): # Load a pre-trained discriminator
        self.d.load_state_dict(path)
    
    def inference(self, img): # Generate output
        f = self.enc.forward(img).cuda()
        return self.g(torch.cat((f, torch.randn((f.shape[0], 512 - f.shape[1], f.shape[2], f.shape[3])).cuda()), 1))
