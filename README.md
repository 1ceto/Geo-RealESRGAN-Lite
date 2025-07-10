# Geo-realESRGAN

Geo-realESRGAN is a GAN-based deep learning model designed for super-resolving 3D reservoir property images under sparse data conditions. It integrates orthogonal slicing and multi-channel encoding strategies to achieve enhanced resolution and geological fidelity.

## 🌟 Features

✅ Improved resolution & fidelity over traditional interpolation  
✅ Adaptable to various geoscience and reservoir datasets

---


## 🔗 Pretrained Model

Download the pretrained model (`Geo-realESRGAN_net_g_50000.pth`) here:

👉 [Baidu Netdisk link](https://pan.baidu.com/s/11jvtr9ij_lCr_erkvLVLTQ) (extraction code: 7ucy)

After downloading, place it into:
```
experiments/RealESRGANx2_finetune_yourdata/models/
```


## 📦 Installation

1️⃣ Clone the repository:
```
git clone https://github.com/yourname/Geo-realESRGAN.git
cd Geo-realESRGAN
```

2️⃣ Install Python dependencies:
```
pip install -r requirements.txt
```

---

## 🚀 Run Inference

Example command to run super-resolution inference:
```
python inference_realesrgan.py -n RealESRGANx2plus -i datasets/val_LR -o results/val_SR_best50000 --model_path experiments/RealESRGANx2_finetune_yourdata/models/net_g_50000.pth -s 2
```

### Parameters:
- `-n` : Model name (default `RealESRGANx2plus`)
- `-i` : Input folder with low-resolution images
- `-o` : Output folder for super-resolved results
- `--model_path` : Path to the pretrained model
- `-s` : Scale factor (e.g., 2)

---

## 📂 Project Structure

```
Geo-realESRGAN/
├── datasets/                   # Input data (optional example files)
│   └── val_LR/
├── experiments/
│   └── RealESRGANx2_finetune_yourdata/
│       └── models/
│           └── net_g_50000.pth
├── inference_realesrgan.py     # Inference script
├── train.py                    # (optional) Training script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 💬 Notes

- Make sure your input images are placed in `datasets/val_LR/`.
- The output results will be saved to `results/val_SR_best50000/` (or your specified `-o` folder).
- If you train new models, update `--model_path` accordingly.

---

## 📧 Contact

If you have any questions, feel free to open an issue or contact the author.
