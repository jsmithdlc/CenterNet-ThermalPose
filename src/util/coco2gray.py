import cv2
import glob
import os
import shutil
import numpy as np
from tqdm import tqdm

dataset_dir = "../coco/train2017/"
img_paths = glob.glob(dataset_dir+"*.jpg")

if os.path.isdir("./train2017"):
	shutil.rmtree("./train2017")
os.mkdir("./train2017")
for idx, img_path in tqdm(enumerate(img_paths)):
	img = cv2.imread(img_path)
	img_name = img_path.split("/")[-1]
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_gray_3_channel = np.zeros((img_gray.shape[0],img_gray.shape[1],3))
	img_gray_3_channel[:,:,0] = img_gray
	img_gray_3_channel[:,:,1] = img_gray
	img_gray_3_channel[:,:,2] = img_gray
	cv2.imwrite("./train2017/{}".format(img_name),img_gray_3_channel)
