# Computer Vision Research Intern Assignment 1

This repository contains assignments for computer vision research interns to gain hands-on experience with modern deep learning architectures, specifically Vision Transformers (ViT) and U-Net-based image generation.

## 🎯 Assignment Overview

The assignment consists of two main parts:
1. Implementing a Vision Transformer (ViT) from scratch for CIFAR-10 classification
2. Modifying an existing repository to implement U-Net architecture for CIFAR-10 image generation

## 🚀 Getting Started

### Prerequisites
- Python programming experience
- PyTorch basics
- Understanding of deep learning fundamentals
- Basic computer vision knowledge

### Environment Setup
```bash
# Create and activate conda environment
conda create -n cv-research python=3.8
conda activate cv-research

# Install required packages
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib numpy tqdm einops tensorboard
```

### Dataset Implementation

Here's a sample implementation of the dataset and dataloader:

```python
# data/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CIFAR10Dataset:
    def __init__(self, root_dir="./data", train=True, image_size=224):
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Add augmentation for training
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=train,
            download=True,
            transform=self.transform
        )

    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=4):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

# Usage example:
def get_dataloaders(batch_size=64, image_size=224):
    train_dataset = CIFAR10Dataset(train=True, image_size=image_size)
    val_dataset = CIFAR10Dataset(train=False, image_size=image_size)
    
    train_loader = train_dataset.get_dataloader(batch_size=batch_size)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
```

Sample usage in training script:
```python
# train.py

from data.dataset import get_dataloaders

# Initialize dataloaders
train_loader, val_loader = get_dataloaders(
    batch_size=64,
    image_size=224  # ViT typically uses 224x224 images
)

# Training loop example
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()  # Move to GPU
        labels = labels.cuda()
        
        # Your training code here
        ...
```

## 📚 Part 1: Vision Transformer Implementation

### Key Resources
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [Phil Wang's implementation](https://github.com/lucidrains/vit-pytorch)
- [Transformer++](https://arxiv.org/abs/2312.00752)

### Implementation Checklist
- [ ] Data preparation (CIFAR-10)
  - [ ] Dataset loading
  - [ ] Data transforms
  - [ ] Data loaders
- [ ] ViT architecture
  - [ ] Patch Embedding
  - [ ] Position Embedding
  - [ ] Multi-head Self-Attention
  - [ ] MLP block
  - [ ] Transformer Encoder
  - [ ] Classification head
- [ ] Training pipeline
  - [ ] Training loop
  - [ ] Validation
  - [ ] Checkpointing
  - [ ] LR scheduling
- [ ] Analysis
  - [ ] Training/validation curves
  - [ ] Attention map visualization
  - [ ] Performance comparison
- [ ] Transformer++
    - [ ] Modify the MLP block
    - [ ] Modify the activation function
    - [ ] Compare with the previous implementation

### Project Structure
```
vit_implementation/
├── data/
│   └── dataset.py
├── model/
│   ├── patch_embed.py
│   ├── attention.py
│   ├── transformer.py
|   ├── transformer++.py
│   ├── mlp.py
│   └── vit.py
├── utils/
│   ├── training.py
│   └── visualization.py
└── train.py
```

## 🎨 Part 2: U-Net Diffusion Image Generation

### Key Resources
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [FastAI U-Net Tutorial](https://www.fast.ai/posts/2021-09-15-advanced-unet.html)

### Recommended Base Repositories
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

### Implementation Steps
1. Repository Setup
   - Fork chosen repository
   - Create new branch
   - Understand existing architecture

2. U-Net Modifications
   - Implement skip connections
   - Modify encoder-decoder
   - Adjust channel dimensions
   - Implement upsampling

3. Training Adaptation
   - Modify training loop
   - Implement loss functions
   - Add logging/visualization
   - Tune hyperparameters

4. Evaluation
   - Generate samples
   - Compare architectures
   - Analyze quality
   - Document changes
   - Understand FID score
   - Understand IS score
   - Draw the relationship between number of denoising steps and FID/IS scores

## 📝 Deliverables

Submit the following:
1. Source Code
   - Well-documented implementation
   - Clear commit history
   - README updates

2. Documentation
   - Implementation details
   - Architecture changes
   - Results analysis
   - Future improvements

3. Results
   - Training logs
   - Model checkpoints
   - Generated samples
   - Visualization plots


## ⏰ Timeline

### Weeks 1-2: ViT Implementation
- Days 1-3: Setup and research
- Days 4-7: Basic implementation
- Days 8-10: Training/debugging
- Days 11-14: Optimization

### Weeks 3-4: U-Net Generation
- Days 1-3: Repository setup
- Days 4-7: Architecture changes
- Days 8-10: Training/debugging
- Days 11-14: Analysis/documentation

## 💡 Tips

### Debugging Strategies
- Start simple, add complexity gradually
- Test components individually
- Use print statements strategically
- Implement unit tests
- Visualize intermediate outputs

### Resource Management
- Start with small models
- Use gradient checkpointing
- Monitor GPU usage
- Clean up unused tensors

## 📚 Additional Resources

### PyTorch
- [Documentation](https://pytorch.org/docs/stable/index.html)
- [Tutorials](https://pytorch.org/tutorials/)

### Vision Transformers
- [ViT Explained](https://theaisummer.com/vision-transformer/)
- [Attention Tutorial](https://jalammar.github.io/illustrated-transformer/)

### Image Generation
- [U-Net Guide](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Tiny Diffusion](https://github.com/tanelp/tiny-diffusion)

## 🆘 Support

1. Check provided resources
2. Use discussion forum
3. Contact supervisor during office hours
4. Document issues/solutions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## ✨ Acknowledgments

- Original ViT paper authors
- U-Net paper authors
- Open source community