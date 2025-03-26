import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from models.generator import Generator
from models.discriminator import Discriminator
from utils import noise, DLoss, GLoss

# Set random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '0'

# Check GPU availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    print("working on gpu!")
else:
    print("No gpu! only cpu ;)")

# Training parameters
train_bs = 128
epochs = 40
noise_dim = 100
LR = 0.0002

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

training_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=training_data, batch_size=train_bs, shuffle=True, drop_last=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()

if is_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        # Prepare labels
        targets_real = torch.FloatTensor(images.size(0), 1).uniform_(0.8, 1.0)
        targets_fake = torch.FloatTensor(images.size(0), 1).uniform_(0.0, 0.2)

        if is_cuda:
            targets_real = targets_real.cuda()
            targets_fake = targets_fake.cuda()
            images = images.cuda()

        # Train Discriminator
        optimizer_D.zero_grad()
        logits_real = discriminator(images)
        fake_images = generator(noise(train_bs, noise_dim, is_cuda)).detach()
        logits_fake = discriminator(fake_images)
        discriminator_loss = DLoss(logits_real, logits_fake, targets_real, targets_fake)
        discriminator_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_images = generator(noise(train_bs, noise_dim, is_cuda))
        logits_fake = discriminator(fake_images)
        generator_loss = GLoss(logits_fake, targets_real)
        generator_loss.backward()
        optimizer_G.step()

    print("Epoch:  ", epoch)
    print("D Loss: ", discriminator_loss.item())
    print("G Loss: ", generator_loss.item())

    if epoch % 2 == 0:
        viz_batch = fake_images.data.cpu().numpy()
        viz_batch = viz_batch[:100,:,:,:]
        viz_batch = viz_batch.reshape(-1,28*28).squeeze()
        viz_batch = viz_batch.reshape(10,10, 28,28).transpose(0,2,1,3).reshape(28*10,-1)

        plt.figure(figsize = (8,8))
        plt.axis('off')
        plt.imshow(viz_batch, cmap='gray')
        plt.savefig(f'outputs/epoch_{epoch}.png')
        plt.close()

print("Training finished!")