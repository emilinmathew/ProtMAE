# ProtMAE: Masked Autoencoder for Protein Distance Map Reconstruction

## Overview
ProtMAE is a self-supervised learning framework designed to reconstruct protein distance maps from masked inputs. Inspired by masked autoencoders in computer vision, ProtMAE leverages a Vision Transformer (ViT) backbone to learn structural representations of proteins efficiently and at scale.  

## Key Features
- **Self-Supervised Pretraining**: Learns protein structure representations without labeled data by reconstructing masked regions of distance maps.  
- **Vision Transformer Backbone**: Adapts transformer-based architectures for protein structure learning.  
- **Downstream Performance**: Demonstrates strong results on:  
  - Secondary structure prediction  
  - Contact map inference  
- **Large-Scale Training**: Pretrained on an 800,000-fragment protein dataset to evaluate structural learning efficiency.  

## Applications
- Protein representation learning  
- Structure-based downstream predictions  
- Computational biology research
