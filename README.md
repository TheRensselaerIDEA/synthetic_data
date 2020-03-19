# synthetic_data 
synthetic_data is a Python module that enables generation of synthetic data from real data. The module enables generation of data which can be distributed easily without revealing private information.

## Dependencies

- Python (>= 3.6)
- NumPy (1.17.0)
- Pandas
- SciPy
- scikit-learn
- tensorflow (1.13.1)
- progress
- psutil
- tqdm
- matplotlib
- seaborn

## Installation

1. Download the repository to your machine using the command below which will generate a folder **synthetic_data**
```
git clone https://github.com/TheRensselaerIDEA/synthetic_data.git
```
2. Go inside the folder **synthetic_data**
```
cd synthetic_data
```
3. Install all dependencies using the command below
```
python3 setup.py install
```

## Usage

The step by step guide to use the package and all its functionalities is available as a Jupyter notebook [here](https://github.com/TheRensselaerIDEA/synthetic_data/blob/master/Package%20Usage%20Guide.ipynb)

## GPU Support

The package also supports the use of GPUs to facilitate the training of the model. To access GPUs, you need to install `tensorflow-gpu==1.13.1` and then define the visible CUDA devices.

### Install tensorflow-gpu

Note that the package is developed on Tensorflow 1.13.1 and thus, it is recommended to use the same version for `tensorflow-gpu` to avoid incompatibility issues.

```
pip install tensorflow-gpu==1.13.1
```

### Set visible devices

Place this code at the start of your Python script. Considering the visible CUDA devices are 2 and 3, the code shall be:
```
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
```
