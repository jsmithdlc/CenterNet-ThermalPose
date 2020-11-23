import json
import os
import glob
import numpy as np
def order_anns_by_id(json_file):
	anns = json_file['annotations']
	#num_imgs = len(json_file['images'])
	ordered_anns = {"ignore":[]}
	for ann in anns:
		if ann['image_id'] in ordered_anns.keys():
			ordered_anns[ann['image_id']].append(ann)
		else:
			ordered_anns[ann['image_id']] = [ann]
	return ordered_anns
	
def renameImages(json_file,bbox_file,set_name,last_img,last_ann,use_maskRCNN):
	ordered_anns = order_anns_by_id(json_file)
	#json_annotations = json_file["annotations"]
	new_anns = []
	new_imgs = []
	ann_counter = 0
	img_counter = 0
	for index, img_info in enumerate(json_file['images']):
		old_filename = img_info['file_name']
		old_id = img_info['id']
		new_filename = "{}_train{}".format(set_name,old_filename)
		new_id = last_img+img_counter
		bboxes = bbox_file[new_filename]

		if old_id not in list(ordered_anns.keys()):
			print("Image:{} does not have annotations".format(old_filename))
			continue

		anns_this_img = ordered_anns[old_id]
		this_anns = []
		success = True

		for ann in anns_this_img:
			this_bboxes = bboxes["bboxes"].copy()
			this_kpts = ann['keypoints'].copy()
			area_id,bbox = find_bbox(this_bboxes,this_kpts,use_maskRCNN)
			if bbox == []:
				success = False
				ann_counter -= len(this_anns)
				break
			ann['bbox'] = bbox
			ann['image_id'] = new_id
			ann['id'] = last_ann + ann_counter
			if use_maskRCNN:
				ann['area'] = bboxes["areas"][area_id]
			else:
				ann['area'] = ann['bbox'][2]*ann['bbox'][3]
			this_anns.append(ann)
			ann_counter += 1

		if(success):
			this_img = json_file['images'][index]
			this_img['file_name'] = new_filename
			this_img['id'] = new_id
			new_anns.extend(this_anns)
			new_imgs.append(this_img)
			img_counter += 1

	json_file['annotations'] = new_anns
	json_file['images'] = new_imgs
	last_img += len(json_file['images'])
	last_ann += len(json_file['annotations'])
	return json_file, last_img, last_ann


def find_bbox(bboxes,kpts,use_maskRCNN):
	if (use_maskRCNN):
		start_kpt = -3
		kpt_ref = kpts[start_kpt:]
		while kpt_ref[-1] == 0:
			kpt_ref = kpts[start_kpt:start_kpt+3]
			start_kpt -= 3
			if start_kpt <= 0:
				return 0, []
		xy_kpt = kpt_ref[:-1]
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
		kpts_x = np.ma.masked_equal(kpts[0::3],0).compressed()
		kpts_y = np.ma.masked_equal(kpts[1::3],0).compressed()
		min_x,max_x = np.min(kpts_x),np.max(kpts_x)
		min_y,max_y = np.min(kpts_y),np.max(kpts_y)
		width = max_x - min_x
		height = max_y - min_y
		return 0,[int(min_x),int(min_y),int(width),int(height)]



def deleteNeck(json_file):
	anns = json_file['annotations']
	new_anns = []
	for ann in anns:
		if len(ann['keypoints']) == 54:
			if ann['keypoints'][-1] > 0:
				ann['num_keypoints'] -= 1
			ann['keypoints'] = ann['keypoints'][:-3]
		new_anns.append(ann)
	json_file['annotations'] = new_anns
	json_file['categories'][0]["keypoints"] = json_file['categories'][0]["keypoints"][:-1]
	return json_file

def completeNeck(json_file):
	anns = json_file['annotations']
	new_anns = []
	for ann in anns:
		if len(ann['keypoints']) == 51:
			ann['keypoints'].extend([0,0,0])
		new_anns.append(ann)
	json_file['annotations'] = new_anns
	json_file['categories'][0]["keypoints"] = json_file['categories'][0]["keypoints"][:-1]
	return json_file

def main(data_path, which_set,delete_neck=False,use_maskRCNN = False):
	json_paths = glob.glob(data_path+"*.json")
	if(len(json_paths)==0):
		print("No json files found at specified directory")
	output_dir= data_path+"joined/"
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	json_files = []
	bbox_jsons = {}
	set_names = []
	for path in json_paths:
		set_name = path.split("/")[-1].split("_")[-1].split(".")[0]
		set_names.append(set_name)
		with open(path) as f:
			json_files.append(json.load(f))
		bbox_path = "{}bbox/bboxes_{}.json".format(data_path,set_name)
		with open(bbox_path) as f:
				bbox_jsons[set_name]=json.load(f)


	last_img_id = 0
	last_ann_id = 0
	new_anns = {"images":[],"annotations":[]}
	for set_name, file in zip(set_names,json_files):
		bbox_file = bbox_jsons[set_name]
		new_json, last_img_id, last_ann_id = renameImages(file,bbox_file,set_name,last_img_id,last_ann_id,use_maskRCNN)
		if delete_neck:
			new_json = deleteNeck(new_json)
		else:
			new_json = completeNeck(new_json)

		new_anns["images"].extend(new_json["images"])
		new_anns["annotations"].extend(new_json["annotations"])
	new_anns["info"] = file["info"]
	new_anns["licenses"] = file["licenses"]
	new_anns["categories"] = file["categories"]
	if not delete_neck:
		new_anns["categories"][0]["skeleton"] = [[16,14],[14,12],[17,15],[15,13],[12,13],[18,12],[18,13],[6,18],[7,18],
		            						[6,8],[7,9],[8,10],[9,11],[1,18],[2,3],[1,2],[1,3],[2,4],[3,5]]
	with open(output_dir + "chenWang_{}.json".format(which_set), "w") as fp:
			json.dump(new_anns, fp)


if __name__ == '__main__':
	main(data_path="../../../ThemalPost-Data/annotations/train/",which_set="train",\
		delete_neck=False,use_maskRCNN=False)

