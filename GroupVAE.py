import copy

import torch
from torch import nn
from torch.nn import functional as F
import Lie.LieAlgebra
import Lie.LieFunction
import Lie.LieGroup


class MyVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims=None):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        modules = []
        if hidden_dims is None:
            # hidden_dims = [32, 64, 128, 256, 512]
            hidden_dims = [512, 256, 128, 64, 32]

        self.hidden_dims = copy.copy(hidden_dims)

        ##############################
        # replace ??? with proper local variables

        in_dim = in_channels
        for h_dim in self.hidden_dims:
            # one convolution layer
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_dim,#???
                              out_channels=h_dim,#???
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        ##############################

        ##############################
        # the central hidden layer of the model.

        # autoencoder version of the representation layer. This is used by default
        self.z_simple = nn.Linear(self.hidden_dims[-1] * 4, latent_dim)

        # VAE Reparametrization Layer
        self.z_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.z_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        ##############################

        # Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh())

    def encode(self, x):
        """Encodes the input into parameters of a normal distribution."""
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)
        ##############################
        # update this along with reparameterize() and forward() to turn this into vae
        # Compute mean and variance of the latent distribution
        # Use mu and var layers we defined in the init

        mu = self.z_mu(z)
        log_var = self.z_var(z)
        z = [mu, log_var]

        ##############################

        return z

    def decode(self, z):
        """Latent space to image space"""
        y = self.decoder_input(z)
        y = y.view(-1, self.hidden_dims[-1], 2, 2)  #
        y = self.decoder(y)
        y = self.final_layer(y)
        return y

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)"""

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(logvar)

        mu1 = Lie.LieAlgebra.AlgebraCreate('so3',[mu[0],mu[1],mu[2]])
        mu2 = Lie.LieAlgebra.AlgebraCreate('so3',[mu[3],mu[4],mu[5]])
        mu3 = Lie.LieAlgebra.AlgebraCreate('so3',[mu[6],mu[7],mu[8]])


        std1 = Lie.LieAlgebra.AlgebraCreate('so3',[std[0]*eps[0],std[1]*eps[1],std[2]*eps[2]])
        std2 = Lie.LieAlgebra.AlgebraCreate('so3',[std[3]*eps[3],std[4]*eps[4],std[5]*eps[5]])
        std3 = Lie.LieAlgebra.AlgebraCreate('so3',[std[6]*eps[6],std[7]*eps[7],std[8]*eps[8]])

        return eps * std + mu

    def forward(self, x):

        ##############################
        # update this along with reparametrize() and encode() to turn this into vae

        # z = self.encode(x)
        # mu = torch.zeros_like(z)
        # log_var = torch.zeros_like(z)

        mu, log_var = self.encode(x) #???
        z = self.reparameterize(mu, log_var)

        ##############################

        return [self.decode(z), x, mu, log_var]

    def loss(self, x, y, z_mu, z_log_var, kl_w):
        """VAE loss
        :param kl_w: Account for the minibatch samples from the dataset"""

        recons_loss = F.mse_loss(y, x)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kl_w * kl_loss
        return loss

    def sample(self, z=None, device='cpu'):
        """Sample image from the latent space."""
        if not z:
            z = torch.randn(1, self.latent_dim).to(device)
        else:
            assert z.shape[1] == self.latent_dim, "z must be of shape [1, {}]".format(self.latent_dim)

        y = torch.clamp(self.decode(z), 0.0, 1.0)
        return y

    def generate(self, x):
        """return the reconstructed image from x"""
        return self.forward(x)[0]


if __name__ == "__main__":
    vae = MyVAE(3, 10)
    x = torch.randn(5, 3, 64, 64)
    y, _, mu, logvar = vae(x)
    loss = vae.loss(y, x, mu, logvar, 1)
    print(loss)