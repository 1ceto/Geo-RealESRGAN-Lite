from model import GeoRealESRGANLite
from utils import load_image, save_image
import torch

if __name__ == '__main__':
    model = GeoRealESRGANLite()
    model.eval()

    img = load_image('example_data/example_LR.png')
    with torch.no_grad():
        out = model(img)
    save_image(out, 'outputs/output_SR.png')
