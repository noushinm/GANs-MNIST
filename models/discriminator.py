import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()
        
        self.layer1 = nn.Linear(input_size, 512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        b_size, n_channels, height, width = x.size()
        x = x.view(b_size, n_channels * height * width)
        
        x = self.layer1(x)
        x = self.leaky_relu(x)
        
        x = self.layer2(x)
        x = self.leaky_relu(x)
        
        y = self.layer3(x)
        return y