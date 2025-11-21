import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

# Simple Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Simple Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Train loop placeholder
def train():
    generator = Generator()
    discriminator = Discriminator()

    dataset = datasets.MNIST(root='data/train', download=True,
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss = nn.BCELoss()
    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(5):
        for img, _ in loader:
            img = img.view(-1, 784)

            z = torch.randn(img.size(0), 128)
            fake = generator(z)

            # Train discriminator
            D_real = discriminator(img)
            D_fake = discriminator(fake.detach())
            loss_D = loss(D_real, torch.ones_like(D_real)) + loss(D_fake, torch.zeros_like(D_fake))

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Train generator
            D_fake = discriminator(fake)
            loss_G = loss(D_fake, torch.ones_like(D_fake))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")

if __name__ == "__main__":
    train()
