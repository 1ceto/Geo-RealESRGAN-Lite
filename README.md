# Geo-RealESRGAN-Lite

Minimal reproducible code for the Geo-RealESRGAN model described in the manuscript:
"Perception-Guided Deep Learning for High-Resolution Reconstruction of Sparse 3D Reservoir Property Models"

##  Structure

- `main.py` - Run super-resolution inference
- `model.py` - Generator model (RRDB-based)
- `utils.py` - Image I/O utilities
- `example_data/` - Input low-resolution image folder
- `outputs/` - Output result folder

##  How to Use

1. Install dependencies:
   ```bash
   pip install torch pillow numpy

