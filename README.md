# chrome-vision

Chrome Vision is a computer vision model
that improves the contrastive learning framework [MoCo’s](https://arxiv.org/abs/1911.05722) accuracy
by detecting and cutting images before classification.

## Prerequisites

### Setup conda environment
#### Create environment

```bash
conda env create -f environment.yml
# conda env remove -n cv
```

#### List environment

```bash
conda env list
```

#### Activate enviroment

```bash
conda activate cv
# conda deactivate
```

### Install PyTorch

#### If using CPU

```bash
conda install cpuonly -c pytorch
```

#### If using GPU with CUDA support

```bash
conda install pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Usage
Run chrome vision inside the `train.ipynb` jupyter notebook
