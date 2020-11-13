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

cpdef tuple nearest(list ref_list,np.ndarray obj_tuple):
	cdef tuple nearest_tuple, iter_tuple
	cdef int min_distance = 1000
	cdef int this_min
	for iter_tuple in ref_list:
		this_min = distance(iter_tuple,obj_tuple)
		if (this_min < min_distance):
			min_distance = this_min
			nearest_tuple = iter_tuple

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

cdef np.ndarray make_cmap():
	cdef np.ndarray[np.uint8_t,ndim=3] img_colorbar,cmap

	img_colorbar = cv2.imread("./colorbar_thermal.jpg")
	cmap = np.zeros((256,1,3),dtype=np.uint8)

	cdef float pointer,step
	cdef int index
	pointer = 0.0
	index = 0
	step = float(img_colorbar.shape[1])/256.0
	while pointer < img_colorbar.shape[1]:
		cmap[255-index,:,:] = np.array(img_colorbar[2,int(pointer),:])
		pointer += step
		index += 1
	return cmap

def main(opts):
	cdef np.ndarray[np.uint8_t,ndim=1] gray_values = np.arange(256,dtype=np.uint8)
	cdef np.ndarray[np.uint8_t,ndim=3] cmap
	cmap = make_cmap()
	color_values = map(tuple,cv2.applyColorMap(gray_values,cmap).reshape(256,3))
	cdef dict color_to_gray_map = dict(zip(color_values, gray_values))
	cdef list paths = glob.glob("{}/*{}".format(opts.data_path,opts.img_format))
	if(not os.path.isdir(opts.output_dir)):
		os.mkdir(opts.output_dir)

	cdef str preamble
	preamble = ""
	if(opts.include_set_name):
		preamble = opts.data_path.split("/")[-1] + "_"

	cdef str img_path, output_path
	cdef np.ndarray[np.uint8_t,ndim=3] color_image 
	cdef np.ndarray[np.uint8_t,ndim=2] gray_img
	cdef np.ndarray[np.uint8_t,ndim=1] bgr
	cdef list color_to_gray_map_keys = list(color_to_gray_map.keys())
	for _,img_path in tqdm(enumerate(paths)):
		output_path = "{}/{}{}".format(opts.output_dir,preamble,img_path.split("/")[-1])
		if os.path.exists(output_path):
			continue
		color_image = cv2.imread(img_path)
		color_image = resize(color_image,10)
		gray_img = np.apply_along_axis(lambda bgr: color_to_gray_map[nearest(color_to_gray_map_keys,bgr)], 2, color_image)
		cv2.imwrite(output_path,gray_img)
