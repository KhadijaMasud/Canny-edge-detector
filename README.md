# Canny Edge Detector 
---

## 📌 Overview
This project is a step-by-step **implementation of the Canny Edge Detector** in Python.  
It processes grayscale images and produces intermediate results (Gaussian smoothing, gradients, magnitude, quantized directions, NMS) and the final Canny edge maps.  
It also includes a script to generate **derivative and double derivative of Gaussian plots** for the mathematical part of the assignment.  

---

## ⚙️ Requirements
- Python 3.8+  
- Required libraries:
  - `numpy`  
  - `Pillow` (PIL)  
  - `matplotlib` (for math plotting only)  

Install dependencies:
```bash
pip install numpy pillow matplotlib
```
---
## Example Run
```bash
python main.py --input_folder sample_images --output_folder Results --input_ext jpg --sigma 1.0
```

