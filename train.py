# train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image

from model import Generator, Discriminator
from perceptual_loss import VGGPerceptualLoss

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, size=64):
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        self.transforms = Compose([Resize((size*4, size*4)), ToTensor()])

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = ToTensor()(Image.open(self.lr_paths[idx]).convert('RGB'))
        hr = self.transforms(Image.open(self.hr_paths[idx]).convert('RGB'))
        return lr, hr

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = Generator().to(device)
    d = Discriminator().to(device)
    perceptual_loss = VGGPerceptualLoss(use_gpu=device.type=='cuda')

    g_opt = optim.Adam(g.parameters(), lr=1e-4)
    d_opt = optim.Adam(d.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    data = SRDataset('data/LR', 'data/HR', size=64)
    loader = DataLoader(data, batch_size=4, shuffle=True)

    for epoch in range(100):
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            # Generator forward
            sr = g(lr)
            # Discriminator loss
            real_logits = d(hr)
            fake_logits = d(sr.detach())
            d_loss = bce_loss(real_logits, torch.ones_like(real_logits)) + \
                     bce_loss(fake_logits, torch.zeros_like(fake_logits))
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Generator loss
            fake_logits2 = d(sr)
            g_adv = bce_loss(fake_logits2, torch.ones_like(fake_logits2))
            g_l1 = l1_loss(sr, hr)
            g_perc = perceptual_loss(sr, hr)
            g_loss = g_l1 + 0.006 * g_perc + 1e-3 * g_adv
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        print(f"Epoch {epoch+1}/100 — D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # Save checkpoints
        os.makedirs('weights', exist_ok=True)
        torch.save(g.state_dict(), 'weights/generator_epoch%d.pth'%epoch)
        torch.save(d.state_dict(), 'weights/discriminator_epoch%d.pth'%epoch)

if __name__ == '__main__':
    train()
