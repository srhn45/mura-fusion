# MURA Fusion

Multi-image patient-level classification for MURA (Musculoskeletal Radiographs) dataset using attention-based fusion of pretrained backbones.

## Overview

This project implements a deep learning model that classifies musculoskeletal X-ray studies as abnormal or normal by:
- Processing multiple images per patient study
- Using pretrained backbones (ResNet50/101/152, ViT) and a customizable ViT as feature extractors
- Fusing image-level features with learnable weighting
- Producing a patient/study-level prediction from multiple images

### Work in progress