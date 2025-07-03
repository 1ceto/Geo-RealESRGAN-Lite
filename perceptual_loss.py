# perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None, use_gpu=False):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleDict({
            'relu1_2': vgg[:4],
            'relu2_2': vgg[4:9],
            'relu3_4': vgg[9:18],
        })
        for param in self.layers.parameters():
            param.requires_grad = False

        self.weights = layer_weights or {'relu1_2': 1.0, 'relu2_2': 1.0, 'relu3_4': 1.0}
        self.transform = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, sr, hr):
        sr = self.transform(sr.squeeze(0)).unsqueeze(0).to(self.device)
        hr = self.transform(hr.squeeze(0)).unsqueeze(0).to(self.device)
        loss = 0.0
        for name, module in self.layers.items():
            sr_feat = module(sr)
            hr_feat = module(hr)
            loss += self.weights[name] * nn.functional.l1_loss(sr_feat, hr_feat)
        return loss
