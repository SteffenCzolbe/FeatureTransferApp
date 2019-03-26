import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module): # 7000 loss
    """
    VAE from paper β-VAE: L EARNING BASIC V ISUAL CONCEPTS WITH A
        CONSTRAINED V ARIATIONAL F RAMEWORK
    """
    def __init__(self, latent_space=32):
        super(BetaVAE, self).__init__() # call super constructor
        
        self.latent_space = latent_space
        
        # (3,64,64)
        self.conv0 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        # (32,32,32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        # (32,16,16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # (64,8,8)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        # (64,4,4)
        self.fc4 = nn.Linear(64*4*4, 256)
        # (256)

        self.mu = nn.Linear(256, latent_space) # mapping from a 784 (28x28) to 400 dim vector
        self.logvar = nn.Linear(256, latent_space)
        self.dec = nn.Linear(latent_space, 256)
        
        # (256)
        self.fc5 = nn.Linear(256, 64*4*4)
        # (64,4,4)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # (64,8,8)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # (32,16,16)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        # (32,32,32)
        self.deconv0 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        # (3,64,64)
        
    def encode(self, x):
        h1 = F.relu(self.conv0(x))
        h2 = F.relu(self.conv1(h1)) 
        h3 = F.relu(self.conv2(h2)) 
        h4 = F.relu(self.conv3(h3)) 
        h5 = F.relu(self.fc4(h4.view(-1, 64*4*4)))
        return self.mu(h5), self.logvar(h5)

    def reparameterize(self, mu, logvar): # reparameterization trick
        std = torch.exp(0.5*logvar) # calculates std
        eps = torch.randn_like(std) # samples an epsilon
        return eps.mul(std).add_(mu) # returns sample as if drawn from mu, std

    def decode(self, z):
        h6 = F.relu(self.dec(z))
        h5 = F.relu(self.fc5(h6))
        h4 = F.relu(self.deconv3(h5.view(-1,64,4,4)))
        h3 = F.relu(self.deconv2(h4))
        h2 = F.relu(self.deconv1(h3))
        h1 = self.deconv0(h2)
        return torch.sigmoid(h1)

    def forward(self, x):  # implements the entire feed-forward network.
        mu, logvar = self.encode(x)  # encode
        z = self.reparameterize(mu, logvar)        # sample latent variable
        return self.decode(z), mu, logvar, z          # decode, return
