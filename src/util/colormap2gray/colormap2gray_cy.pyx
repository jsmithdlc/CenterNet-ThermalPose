import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np

import cmapy
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap


cpdef int distance((int,int,int) ref_tuple,np.ndarray obj_tuple):
	cdef int d1 = abs(int(ref_tuple[0]) - int(obj_tuple[0]))
	cdef int d2 = abs(int(ref_tuple[1]) - int(obj_tuple[1]))
	cdef int d3 = abs(int(ref_tuple[2]) - int(obj_tuple[2]))
	return d1 + d2 + d3

def nearest(list ref_list,np.ndarray obj_tuple):
	nearest_tuple = min(ref_list,key=lambda x: distance(x,obj_tuple))
	return nearest_tuple

cpdef np.ndarray resize(np.ndarray img,int scale):
	cdef int width,height
	cdef (int,int) dim
	cdef np.ndarray[np.uint8_t,ndim=3] resized
	width = int(img.shape[1] * scale / 100)
	height = int(img.shape[0] * scale/ 100)
	dim = (width, height) 
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
	return resized

def main(opts):
	cdef np.ndarray gray_values = np.arange(256,dtype=np.uint8)
	color_values = map(tuple,cv2.applyColorMap(gray_values,cmapy.cmap('nipy_spectral')).reshape(256,3))
	cdef dict color_to_gray_map = dict(zip(color_values, gray_values))
	cdef list paths = glob.glob("{}/*{}".format(opts.data_path,opts.img_format))
	if(not os.path.isdir(opts.output_dir)):
		os.mkdir(opts.output_dir)
	preamble = ""
	if(opts.include_set_name):
		preamble = opts.data_path.split("/")[-1] + "_"

	cdef str img_path
	cdef np.ndarray[np.uint8_t,ndim=3] color_image 
	cdef np.ndarray[np.uint8_t,ndim=2] gray_img
	cdef np.ndarray[np.uint8_t,ndim=1] bgr
	for _,img_path in tqdm(enumerate(paths)):
		color_image = cv2.imread(img_path)
		color_image = resize(color_image,10)
		#gray_img = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
		gray_img = np.apply_along_axis(lambda bgr: color_to_gray_map[nearest(list(color_to_gray_map.keys()),bgr)], 2, color_image)
		cv2.imwrite("{}/{}{}".format(opts.output_dir,preamble,img_path.split("/")[-1]),gray_img)
