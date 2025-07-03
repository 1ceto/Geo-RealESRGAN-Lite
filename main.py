import torch
from PIL import Image
import torchvision.transforms as T
from model import Generator  # 确保 model.py 文件名一致
import os

# === 图像读取与预处理 ===
def load_image(path):
    image = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # [1, C, H, W]

# === 图像保存 ===
def save_image(tensor, path):
    tensor = tensor.clamp(0, 1).squeeze(0)
    image = T.ToPILImage()(tensor.cpu())
    image.save(path)

# === 主函数 ===
def main():
    # 加载模型
    model = Generator()
    model.load_state_dict(torch.load('weights/generator.pth', map_location='cpu'))  # 替换路径
    model.eval()

    # 输入图像路径
    input_path = 'data/example_LR.png'
    output_path = 'outputs/example_SR.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取图像
    img = load_image(input_path)

    # 推理
    with torch.no_grad():
        sr = model(img)

    # 保存结果
    save_image(sr, output_path)
    print(f"✅ Output saved to {output_path}")

if __name__ == '__main__':
    main()
