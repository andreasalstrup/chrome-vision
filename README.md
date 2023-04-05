# chrome-vision

(<a href="https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux/">Install python 3.11 on Ubuntu</a>)

### 1. Create environment

```bash
conda env create -f environment.yml
# conda env remove -n cv
```

### 2. List environment

```bash
conda env list
```

### 3. Activate enviroment

```bash
conda activate cv
# conda deactivate
```

### 4. Install PyTorch

#### 4.1 If using CPU

```bash
conda install cpuonly -c pytorch
```

#### 4.2 If using GPU with CUDA support

```bash
conda install pytorch-cuda=11.6 -c pytorch -c nvidia
```

### 5. Run

```bash
python object_detection.py
```

---
The data folder is in .gitignore so download cityscapes yourself ;)
To create an index file go into the folder in the terminal and use the following command: "ls >> index.csv"
