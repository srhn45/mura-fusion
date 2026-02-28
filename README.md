# MURA Fusion

Multi-image patient-level classification for MURA (Musculoskeletal Radiographs) dataset using attention-based fusion of pretrained backbones.

## Overview

This project implements a deep learning model that classifies musculoskeletal X-ray studies as abnormal or normal by:
- Processing multiple images per patient study
- Using pretrained backbones (ResNet50/101/152, ViT) and a customizable ViT as feature extractors
- Fusing image-level features with learnable weighting
- Producing a patient/study-level prediction from multiple images

## Citation

[MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs](https://arxiv.org/abs/1712.06957) (Rajpurkar et al., 2017)

### Work in progress