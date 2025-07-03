from PIL import Image
import numpy as np
import torch

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((128, 128))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def save_image(tensor, path):
    img = tensor.squeeze().detach().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
