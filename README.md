# Geo-RealESRGAN-Lite

Minimal reproducible code for the Geo-RealESRGAN model described in the manuscript:
"Perception-Guided Deep Learning for High-Resolution Reconstruction of Sparse 3D Reservoir Property Models"

##  Structure

- `main.py` - Run super-resolution inference
- `model.py` - Generator model (RRDB-based)
- `utils.py` - Image I/O utilities

### 1. Install dependencies
```bash
pip install torch torchvision pillow numpy
```

### 2. Prepare your data
- Place your low-resolution image (e.g., `example_LR.png`) in the `data/` folder.
- Optionally, prepare a matching high-resolution image in `data/HR/` for evaluation.

### 3. Download or train weights
- Place pretrained weights at: `weights/generator.pth`
- Or train your own model using:
```bash
python train.py
```

### 4. Run inference
```bash
python main.py
```
The super-resolved output will be saved in the `outputs/` folder.

### 5. Optional: Train your own model
- Place training LR/HR images in `data/LR/` and `data/HR/` folders respectively
- Start training:
```bash
python train.py
```
