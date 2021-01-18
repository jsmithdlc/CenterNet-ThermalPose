# Thermal Pose Estimation Based on CenterNet
Pose Estimation on Thermal Images, based on center point detection by CenterNet

## Project Structure
- **CenterNet**: forked repository from https://github.com/xingyizhou/CenterNet.git, with changes for incorporating HRNet backbone, freezing specific layers and backbones, among other.
- **figures**: includes training plots for different experiments, network graphs for visualizing, output keypoint detection images and captured lab images

## Manually Annotated Thermal Image Dataset
For model finetuning, 600 images were labeled for training and 200 images for testing. Dataset is available at:
https://drive.google.com/drive/folders/1YV7g563ZGlGO-9wx9G0vt6r7itJE8GDZ?usp=sharing

## Setting Up
Code tested on Ubuntu 16.04, with Python 3.6 and Pytorch 1.1. NVIDIA GPUs required for training and testing. Steps to follow:

1. Create Conda Environment
~~~
conda create --name CenterNet python=3.6
conda activate CenterNet
~~~

2. Install Pytorch 1.1, for CUDA version available on pc. For example, for CUDA 10.0, execute:
~~~
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
~~~
And disable cudnn batch normalization. For this, manually open `torch/nn/functional.py`, find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. Usually, this file can be found at `~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/`


3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

~~~
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    python setup.py install --user
 ~~~
 
 4. Clone this repo:
~~~
git clone https://github.com/jsmithdlc/CenterNet-Thermal-Human-Pose-Estimation.git
cd CenterNet-Thermal-Human-Pose-Estimation
~~~

5. Install additional requirements:
~~~
pip install -r requirements.txt
~~~

6. clone and build original DCN2
~~~
cd CenterNet/src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2

vim cuda/dcn_va_cuda.cu
"""
# extern THCState *state;
THCState *state = at::globalContext().lazyInitCUDA();
"""
python setup.py build develop
~~~

7. Compile NMS. Before doing this, comment the following: `#extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]` in setup.py

~~~
cd CenterNet/src/lib/external
python setup.py build_ext --inplace
~~~

8. Download pertained models, from  and move them to `CenterNet/models/`

9. Download thermal images dataset, from https://drive.google.com/drive/folders/1YV7g563ZGlGO-9wx9G0vt6r7itJE8GDZ?usp=sharing, and move it to `CenterNet/data/
