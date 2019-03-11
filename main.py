import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms

batch_size = 64
epoch_num = 300
device = 'cuda' #cpu

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Linear(100, 128)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(128, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # x = input.view(input.size(0), -1)
        x = self.layer1(input)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.layer1(input)
        x = self.relu(x)
        x = self.layer2(x)
        return x
        # x = self.sigmoid(x)
        # return x

D = Discriminator().to(device)
G = Generator().to(device)

G_solver = optim.Adam(G.parameters(), lr=1e-3)
D_solver = optim.Adam(D.parameters(), lr=1e-3)

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, drop_last=True, shuffle=True)

ones_label = torch.ones(batch_size, 1).to(device)
# zeros_label = torch.zeros(batch_size, 1)

for epoch in range(epoch_num):
    total_iter = len(train_loader)
    for batch_idx, (X, _) in enumerate(train_loader):
        X = X.view(batch_size, -1).to(device)

        #Discriminator
        z = torch.randn(batch_size, 100).to(device)
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = f.binary_cross_entropy_with_logits(D_real - D_fake, ones_label)

        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()


        #Generator
        z = torch.randn(batch_size, 100).to(device)
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        G_loss = f.binary_cross_entropy_with_logits(D_fake - D_real, ones_label)

        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()



        if batch_idx % 10 == 0:
            print('epoch {} [{}/{}], D_loss {:.5f}, G_loss {:.5f}'.format(epoch, batch_idx, total_iter, float(D_loss), float(G_loss)))
        if batch_idx % 500 == 0:
            G_sample = G(torch.randn(1, 100).to(device))
            raw_img = G_sample.view(-1, 28).detach().cpu().numpy()
            img = (raw_img * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(img).save('./results/tmp/epoch-{}-iter-{}.png'.format(epoch, batch_idx))



