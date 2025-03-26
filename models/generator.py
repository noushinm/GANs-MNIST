import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, out_size=784):
        super(Generator, self).__init__()
        
        self.layer1 = nn.Linear(noise_dim, 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, out_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        b_size, n_channels = x.size()
        
        x = self.layer1(x)
        x = self.leaky_relu(x)
        
        x = self.layer2(x)
        x = self.leaky_relu(x)
        
        x = self.layer3(x)
        x = self.leaky_relu(x)
        
        x = self.layer4(x)
        x = self.tanh(x)
        
        x = x.view(b_size, 1, 28, 28)
        return x