import sys
CENTERNET_LIB_PATH = "../CenterNet/src/lib/"
CENTERNET_SRC_PATH = "../CenterNet/src/"
sys.path.insert(0, CENTERNET_LIB_PATH)
sys.path.insert(1, CENTERNET_SRC_PATH)

import os
import cv2
import numpy as np
from opts import opts
from detectors.detector_factory import detector_factory
import argparse

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

color_list = np.array(
        [
            1.000, 1.000, 1.000]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
theme = "normal"
colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
			(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255)]

ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
		(255, 0, 0), (0, 0, 255), (255, 0, 255),
		(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
		(255, 0, 0), (0, 0, 255), (255, 0, 255),
		(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]

edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
		[3, 5], [4, 6], [5, 6], 
		[5, 7], [7, 9], [6, 8], [8, 10], 
		[5, 11], [6, 12], [11, 12], 
		[11, 13], [13, 15], [12, 14], [14, 16]]

MODEL_PATH = "../CenterNet/models/multi_pose_dla_3x_gray_384_0frz.pth"
TASK = 'multi_pose' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
detector.pause = False

def add_coco_bbox(img,bbox, cat, conf=1, show_txt=True, img_id='default'):
	bbox = np.array(bbox, dtype=np.int32)
	# cat = (int(cat) + 1) % 80
	cat = int(cat)
	# print('cat', cat, self.names[cat])
	#c = colors[cat][0][0].tolist()
	c = [0,255,0]
	if theme == 'white':
		c = (255 - np.array(c)).tolist()
	txt = '{}{:.1f}'.format("person", conf)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
	cv2.rectangle(
		img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
	if show_txt:
	  cv2.rectangle(img,
	                (bbox[0], bbox[1] - cat_size[1] - 2),
	                (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
	  cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
	              font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

	return img

def add_coco_hp(img,points, img_id='default'): 
	points = np.array(points, dtype=np.int32).reshape(17, 2)
	for j in range(17):
		cv2.circle(img,
				(points[j, 0], points[j, 1]), 3, colors_hp[j], -1)
	for j, e in enumerate(edges):
		if points[e].min() > 0:
			cv2.line(img, (points[e[0], 0], points[e[0], 1]),
			(points[e[1], 0], points[e[1], 1]), ec[j], 2,
			lineType=cv2.LINE_AA)
	return img


def draw_detection(img,results):
	for bbox in results[1]:
		if bbox[4] > opt.vis_thresh:
			ret_img = add_coco_bbox(img,bbox[:4], 0, bbox[4], img_id='multi_pose')
			ret_img = add_coco_hp(ret_img,bbox[5:39], img_id='multi_pose')
		else:
			ret_img = img
	return ret_img

def main(args):
	if args.demo == 'webcam' or \
		args.demo[args.demo.rfind('.') + 1:].lower() in video_ext:
		cam = cv2.VideoCapture(0 if args.demo == 'webcam' else args.demo)
		while True:
			_, img = cam.read()
			cv2.imshow('entrada', img)
			ret = detector.run(img)
			ret_img = draw_detection(img,ret["results"])
			cv2.imshow('deteccion',ret_img)
			time_str = ''
			for stat in time_stats:
				time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
			print(time_str)
			if cv2.waitKey(0 if args.pause else 1) == 27:
				import sys
				sys.exit(0)

	else:
		if os.path.isdir(args.demo):
			image_names = []
			ls = os.listdir(args.demo)
			print(args.demo)
			for file_name in sorted(ls):
				ext = file_name[file_name.rfind('.') + 1:].lower()
				if ext in image_ext:
			 		image_names.append(os.path.join(args.demo, file_name))
		else:
			image_names = [args.demo]

		for (image_name) in image_names:
			img = cv2.imread(image_name)
			cv2.imshow('entrada', img)
			ret = detector.run(img)
			ret_img = draw_detection(img,ret["results"])
			cv2.imshow('deteccion',ret_img)
			if cv2.waitKey(0 if args.pause else 1) == 27:
				import sys
				sys.exit(0)
			time_str = ''
			for stat in time_stats:
				time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
			print(time_str)
			cv2.imshow('entrada', img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--demo', default='', 
                         help='path to image/ image folders/ video. ')
	parser.add_argument('--pause', action='store_true', 
                         help='whether to pause between detections')
	args = parser.parse_args()
	main(args)
