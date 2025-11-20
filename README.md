# Medical Image Super-Resolution & Denoising

Summer internship project at **Genoray FlexLab AI Team**, focusing on image restoration for medical modalities including dental X-ray, mammography, and spine MRI.  
Super-resolution and denoising performed to enhance low-quality medical scans and improve downstream analysis.

---

## Overview
Image restoration applied to:
- Dental / cephalometric imaging  
- Mammography  
- Spine MRI  
- Other radiological images  

Tasks include:
- Super-resolution (×2 / ×4)  
- Gaussian and real-noise denoising  
- Model fine-tuning on medical datasets  

---

## Model References
Implementation inspired by:

- **SwinIR**  
  https://github.com/JingyunLiang/SwinIR

- **KAIR**  
  https://github.com/cszn/KAIR

Backbone structures, training strategies, and restoration pipelines referenced and adapted for medical imaging.

---

## Features
- Transformer-based SR and DN models  
- Classical CNN-based restoration baselines  
- Medical-domain preprocessing and augmentation  
- PyTorch-based training and inference scripts  
- Support for DICOM / PNG datasets  
