# GitHub Tutorial - Getting Started with the Project

This guide will walk you through setting up and working with the Computer Vision Research Assignment using Git and GitHub.

## Prerequisites

1. Install Git:
   - Windows: Download from [git-scm.com](https://git-scm.com/download/windows)
   - Mac: `brew install git`
   - Ubuntu: `sudo apt-get install git`

2. Create a GitHub account at [github.com](https://github.com)

3. Configure Git with your credentials:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step-by-Step Guide

### 1. Fork the Repository
1. Go to the project repository on GitHub
2. Click the "Fork" button in the top-right corner
3. Select your account as the destination

### 2. Clone Your Fork
```bash
# Replace YOUR-USERNAME with your GitHub username
git clone https://github.com/andypinxinliu/CVResearchIntern.git
cd CVResearchIntern
```

### 3. Create Development Branch
```bash
# Create and switch to a new branch
git checkout -b development

# Verify you're on the new branch
git branch
```

### 4. Project Setup
```bash
# Set up conda environment as described in README
conda create -n cv-research python=3.8
conda activate cv-research

# Install CUDA toolkit and nvcc
conda install -c conda-forge cuda-toolkit=11.8 cuda-nvcc=11.8

# Install PyTorch and other requirements
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib numpy tqdm einops tensorboard
```

### 5. Creating Project Structure
```bash
# Create project directories
mkdir -p vit_implementation/{data,model,utils}
touch vit_implementation/data/dataset.py
touch vit_implementation/model/{patch_embed.py,attention.py,transformer.py,transformer++.py,mlp.py,vit.py}
touch vit_implementation/utils/{training.py,visualization.py}
touch vit_implementation/train.py
```

### 6. Regular Development Workflow

#### a. Before starting work
```bash
# Get latest changes from main repository
git remote add upstream https://github.com/andypinxinliu/CVResearchIntern.git
git fetch upstream
git merge upstream/main
```

#### b. Making changes
```bash
# Make your changes to the code
# Stage changes
git add .

# Commit changes with meaningful message
git commit -m "Implemented patch embedding layer"
```

#### c. Pushing changes
```bash
# Push to your fork
git push origin development
```

### 7. Creating Pull Requests
1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your development branch
4. Add description of your changes
5. Submit the pull request

## Best Practices

### Commit Messages
Write clear, concise commit messages:
```bash
# Good examples
git commit -m "Add patch embedding implementation"
git commit -m "Fix attention mask calculation bug"
git commit -m "Improve training loop performance"

# Bad examples
git commit -m "updates"
git commit -m "fix"
git commit -m "wip"
```

### Branch Management
```bash
# List all branches
git branch

# Switch branches
git checkout branch-name

# Delete branch (after merging)
git branch -d branch-name
```

### Keeping Fork Updated
```bash
# Update your fork with original repository changes
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Common Issues and Solutions

### 1. Merge Conflicts
If you get merge conflicts:
```bash
# See which files have conflicts
git status

# Resolve conflicts in each file
# Look for <<<<<<< HEAD markers
# Choose which changes to keep
# Remove conflict markers

# After resolving
git add .
git commit -m "Resolve merge conflicts"
git push origin development
```

### 2. Wrong Branch
If you committed to wrong branch:
```bash
# Save your changes
git stash

# Switch to correct branch
git checkout correct-branch

# Apply changes
git stash pop
```

### 3. Undo Last Commit
If you need to undo last commit:
```bash
# Undo commit but keep changes
git reset --soft HEAD^

# Undo commit and discard changes (be careful!)
git reset --hard HEAD^
```

## Using GitHub Features

### 1. Issues
- Use issues to track bugs and features
- Include clear descriptions and steps to reproduce
- Add labels for better organization

### 2. Pull Requests
- Reference related issues in PRs
- Add detailed descriptions of changes
- Request reviews from maintainers
- Respond to review comments

### 3. Project Boards
- Track progress using project boards
- Organize tasks into columns
- Move cards as work progresses

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [GitHub CLI](https://cli.github.com/)
- [GitHub Desktop](https://desktop.github.com/) (GUI alternative)

## Getting Help

If you encounter issues:
1. Check the Git documentation
2. Search GitHub issues
3. Ask on project discussions
4. Contact your supervisor

Remember to always create backups of important work and test changes in a development branch before merging to main.