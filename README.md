# chrome-vision

chrome-vision is an object detection and image classification model. 

chrome-vision is our bachelor project at AAU (Aalborg University), with inspiration from <a href="https://github.com/facebookresearch/moco">FacebookResearch - MoCo</a>

## Getting started (prerequisites)
<a href="https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux/">Python 3.11 on Ubuntu</a>

<a href="https://pytorch.org/get-started/locally/">Pytorch with CUDA</a>  

After the necessary software has been installed, fork the repository using git, and use the commands below to create a working environment.

1. Create environment
```bash
conda env create -f environment.yml
# conda env remove -n cv
```
2. List environment
```bash
conda env list
# confirm that base and cv are installed
```
3. Activate enviroment
```bash
conda activate cv
# conda deactivate
```
