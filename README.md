# Thermal Pose Estimation Based on CenterNet
<!-- Pose Estimation on Thermal Images, based on center point detection by CenterNet. Example detection images: -->

This repository contains the code and dataset from the following paper:

`J. Smith, P. Loncomilla and J. Ruiz-del-Solar, "Human Pose Estimation using Thermal Images," in IEEE Access, doi: 10.1109/ACCESS.2023.3264714.`

<b> Note that both using the code or using the dataset requires to reference this paper. </b>

Example detection images:

<p align="center"> <img src='figures/samples/sample1.png' align="center" height="230px"> <img src='figures/samples/sample2.png' align="center" height="230px">  <img src='figures/samples/sample3.png' align="center" height="230px"> <img src='figures/samples/sample4.png' align="center" height="230px">                    <img src='figures/samples/sample5.png' align="center" height="230px"> <img src='figures/samples/sample6.png' align="center" height="230px">                    <img src='figures/samples/sample7.png' align="center" height="230px"></p>



## Project Structure
- **CenterNet**: forked repository from https://github.com/xingyizhou/CenterNet.git, with changes for incorporating HRNet & 4-Stack Hourglass backbone, freezing specific layers and backbones, among other.
- **figures**: includes training plots for different experiments, network graphs for visualizing, output keypoint detection images and captured lab images
- **src**: contains program `detect_people.py` for detecting human pose over input image or video and some utilities used during the project development.

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

8. Download pertained models, from https://drive.google.com/drive/folders/1cwINRJrUTxqJ0JMRza9WIGC0EKGuwswy?usp=sharing  and move them to `CenterNet/models/`

9. Download thermal images dataset, from https://drive.google.com/drive/folders/1YV7g563ZGlGO-9wx9G0vt6r7itJE8GDZ?usp=sharing, and move it to `CenterNet/data/`

## Running Demo
Demo program can be run using the following command, using CenterNet DLA architecture for detection:
~~~
cd src
python detect_people.py --demo /path/to/image/or/folder/or/video/ 
~~~
For using Hourglass or HRNet backbones, include parameter `-- arch` with `hourglass` or `hrnet`. Detection can be paused between images using `--pause`. FPS can be displayed using `--show_fps`. To turn off visualizations during detection, use `--visualize 0`

Images with detections can be saved using `--save_img`. Likewise, csv files with detections can be generated using `--save_csv`. In each case, user must specify output directory using `--output_dir /path/to/output/directory/`.

## Training Models
Models can be trained using the following sample command:
~~~
cd CenterNet/src
python main.py multi_pose --exp_id <experiment_name> --dataset thermal_pose --master_batch 8 --batch_size 16 --lr 5e-4 --load_model /path/to/pretrained/model/ --gpus 0,1 --num_epochs 50 --lr_step 35,45
~~~
To change backbone network, specify `--arch` with `hourglass` or `hrnet32`. Experiments with specific layer freezing can be turned on using `--freeze_blocks` and specifying modules to be freezed. For example, to finetune CenterNet DLA with first convolutional block freezed, run above command including `--freeze_blocks base_layer`. Entire backbone network can be freezed simply using `--freeze_backbone`. More details about different possible options can be found at `CenterNet/src/lib/opts.py`.

## Testing Models
Models can be tested using the following sample command:
~~~
cd CenterNet/src
python test.py multi_pose --exp_id hg_3x --dataset thermal_pose --keep_res --load_model /path/to/model/
~~~
You can specify testing resolution using `--input_res <resolution>`. For example, to test over 384 X 384 images, use `--input_res 384`.

## References
    @inproceedings{zhou2019objects,
          title={Objects as Points},
          author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
          booktitle={arXiv preprint arXiv:1904.07850},
          year={2019}
        }
    @article{NewellYD16,
          title={Stacked Hourglass Networks for Human Pose Estimation},
          author={Alejandro Newell and Kaiyu Yang and Jia Deng},
          journal={CoRR},
          url={http://arxiv.org/abs/1603.06937},
          year={2016},
        }
    @inproceedings{SunXLW19,
          title={Deep High-Resolution Representation Learning for Human Pose Estimation},
          author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
          booktitle={CVPR},
          year={2019}
        }
    @article{DBLP:journals/corr/YuWD17,
          title={Deep Layer Aggregation},
          author={Fisher Yu and Dequan Wang and Trevor Darrell},
          journal={CoRR},
          url={http://arxiv.org/abs/1707.06484},
          year={2017},
        }

       

    




