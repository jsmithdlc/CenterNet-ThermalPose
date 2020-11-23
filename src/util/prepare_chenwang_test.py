import glob
import json
from collections import OrderedDict
from datetime import date
import argparse
import numpy as np

class ChenWang:
	def __init__(self, input_dir):
		json_paths = glob.glob(input_dir+"*.json")
		if(len(json_paths)==0):
			print("No json files found at specified directory")
		self.json_files = []
		self.bbox_jsons = {}
		self.set_names = []
		for path in json_paths:
			set_name = path.split("/")[-1].split("_")[-1].split(".")[0]
			self.set_names.append(set_name)
			with open(path) as f:
				self.json_files.append(json.load(f))
			bbox_path = "{}bbox/bbox_{}.json".format(input_dir,set_name)
			with open(bbox_path) as f:
				self.bbox_jsons[set_name]=json.load(f)

	def add_info(self):
		self.info = {"description": "COCO Dataset in Thermal Imaging",
					 "url":"",
					 "version":"1.0",
					 "year":2018,
					 "contributor":"National Sun Yat-Sen University",
					 "date_created":"2018/11/07"}

	def add_licenses(self):
		self.licenses = [[{"url":"none","id":1,"name":"License"}]]

	def add_anns(self):
		self.images = {}
		self.annotations = []
		img_count = 0
		ann_count = 0
		banned_imgs = []
		for i, json_file in enumerate(self.json_files):
			bboxes = self.bbox_jsons[self.set_names[i]]
			for index, img_ann in enumerate(json_file):
				old_filename = img_ann['jpg']
				if self.set_names[i] == "general":
					new_filename = "{}_train{}".format(self.set_names[i],old_filename)
				else:
					new_filename = "{}_{}".format(self.set_names[i],old_filename)
				im_bbox = bboxes[new_filename.split("train")[-1]]
				self.annotations.append({"segmentation":[[]],
										"bbox":im_bbox["bboxes"],
										"area":im_bbox["areas"],
										"num_keypoints":len(img_ann["loc"]),
										"iscrowd":0,
										"keypoints":img_ann["loc"],
										"keypoints_labels":img_ann["label"],
										"image_id":img_count,
										"category_id":1,
										"id":ann_count})
				if not img_count in self.images.keys():
					self.images[img_count]={"license":1,
											 "file_name":new_filename,
											 "coco_url":"",
											 "height":525,
											 "width":600,
											 "date_captured":"",
											 "flickr_url":"",
											 "id":img_count}
					img_count += 1
				ann_count += 1

	def clean_anns(self,use_maskRCNN):
		keypoints = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
					 "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
					 "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","neck"]
		new_annotations = []
		banned_imgs = []
		for ann_id, ann in enumerate(self.annotations):
			img_id = ann['image_id']
			if img_id in banned_imgs:
				continue
			kpts_dict = {kpt:[0,0,0] for kpt in keypoints}
			kpts_xy = ann["keypoints"]
			kpts_labels = ann["keypoints_labels"]
			for i, label in enumerate(kpts_labels):
				xy = kpts_xy[i] + [1]
				kpts_dict[label] = xy
			bbox_id, bbox = self.find_bbox(ann["bbox"],kpts_dict, use_maskRCNN)
			if len(bbox)==0:
				banned_imgs.append(img_id)
				del self.images[img_id]
				continue
			new_kpts = []
			for kpt_label in keypoints:
				new_kpts.extend(kpts_dict[kpt_label])
			ann["keypoints"] = new_kpts		
			ann["bbox"]	= bbox
			if use_maskRCNN:
				ann["area"] = ann["area"][bbox_id]
			else:
				ann["area"] = int(ann["bbox"][2]*ann["bbox"][3])
			del ann["keypoints_labels"]
			new_annotations.append(ann)
		self.annotations = new_annotations
	

	def find_bbox(self, bboxes,kpts_dict, use_maskRCNN):
		if (use_maskRCNN):
			kpt_ref = kpts_dict["neck"]
			if kpt_ref[-1] == 0:
				kpt_ref = kpts_dict["left_hip"]
			xy_kpt = [int(kpt_ref[:-1][0]),int(kpt_ref[:-1][1])]
			for bbox_id, bbox in enumerate(bboxes):
				bbox_x1 = int(bbox[1])
				bbox_y1 = int(bbox[0])
				bbox_x2 = int(bbox[3])
				bbox_y2 = int(bbox[2])
				if (xy_kpt[0] >= bbox_x1) and (xy_kpt[0] <= bbox_x2) and \
				   (xy_kpt[1] >= bbox_y1) and (xy_kpt[1] <= bbox_y2):
				   return bbox_id,[bbox_x1,bbox_y1,bbox_x2-bbox_x1,bbox_y2-bbox_y1] 
			return 0,[]
		else:
			kpts = np.array(list(kpts_dict.values()))
			kpts = kpts[~np.all(kpts == 0, axis=1)]
			min_x, min_y,_ = np.min(kpts,axis=0)
			max_x, max_y,_ = np.max(kpts,axis=0)
			width = max_x - min_x
			height = max_y - min_y
			return 0,[int(min_x),int(min_y),int(width),int(height)]
		

	def deleteNeck(self):
		new_anns = []
		for ann in self.annotations:
			if len(ann['keypoints']) == 54:
				if ann['keypoints'][-1] > 0:
					ann['num_keypoints'] -= 1
				ann['keypoints'] = ann['keypoints'][:-3]
			new_anns.append(ann)
		self.annotations = new_anns

	def completeNeck(self):
		new_anns = []
		for ann in self.annotations:
			if len(ann['keypoints']) == 51:
				ann['keypoints'].extend([0,0,0])
			new_anns.append(ann)
		self.annotations = new_anns


	def add_categories(self, delete_neck=False):
			keypoints = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
						 "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
						 "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
			if not delete_neck:
				keypoints += ["Neck"]
				skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[18,12],[18,13],[6,18],[7,18],
		            		[6,8],[7,9],[8,10],[9,11],[1,18],[2,3],[1,2],[1,3],[2,4],[3,5]]
			else:
				skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
			        		[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
			self.categories = [{"supercategory":"person",
								"id":1,
								"name":"person",
								"keypoints":keypoints,
								"skeleton":skeleton}]


	def transform2coco(self,output_dir,delete_neck=False,use_maskRCNN = False):
		self.add_info()
		self.add_licenses()
		self.add_anns()
		self.clean_anns(use_maskRCNN)
		if delete_neck:
			self.deleteNeck()
		else:
			self.completeNeck()
		self.add_categories(delete_neck)
		print(self.categories)

		output_ann_file = {"info":self.info,
						   "licencese":self.licenses,
						   "images":list(self.images.values()),
						   "annotations":self.annotations,
						   "categories":self.categories}
		with open(output_dir + "chenWang_test.json", "w") as fp:
					json.dump(output_ann_file, fp)


if __name__ == '__main__':
	chenWang = ChenWang("/home/javier/Javier/Universidad/memoria/repositorios/ThemalPost-Data/annotations/val/")
	output_dir = "/home/javier/Javier/Universidad/memoria/repositorios/ThemalPost-Data/annotations/val/joined/"
	chenWang.transform2coco(output_dir=output_dir,delete_neck=True)

