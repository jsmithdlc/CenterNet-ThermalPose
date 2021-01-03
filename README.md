# Thermal Pose Estimation Based on CenterNet
Pose Estimation on Thermal Images, based on center point detection by CenterNet

## Project Structure
- **CenterNet**: forked repository from https://github.com/xingyizhou/CenterNet.git, with changes for incorporating HRNet backbone, freezing specific layers and backbones, among other.
- **figures**: includes training plots for different experiments, network graphs for visualizing, output keypoint detection images and captured lab images

## Manually Annotated Thermal Image Dataset
For model finetuning, 600 images were labeled for training and 200 images for testing. Dataset is available at:
https://drive.google.com/drive/folders/1YV7g563ZGlGO-9wx9G0vt6r7itJE8GDZ?usp=sharing
