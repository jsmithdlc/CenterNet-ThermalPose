import glob
import json
from collections import OrderedDict
from datetime import date
import argparse


class PoseThermal:
    def __init__(self, dir_path):
        json_paths = glob.glob(dir_path+"*.json")
        if(len(json_paths)==0):
            print("No json files found at specified directory")
        self.json_files = []
        for path in json_paths:
            with open(path) as f:
                self.json_files.append(json.load(f))


    def add_info(self):
        self.info = {"description": "Pose Thermal 2020 {} Dataset".format(args.set),
                     "url":"",
                     "version":"1.0",
                     "year":2020,
                     "contributor":"Javier Smith",
                     "date_created":"{}".format(date.today().strftime("%d/%m/%Y"))}

    def add_licenses(self):
        self.licenses = [{}]

    def add_images(self):
        self.images = {}
        for img_idx, ann_file in enumerate(self.json_files):
            self.images[img_idx] = {"license":0,
                                     "file_name":ann_file["imagePath"],
                                     "coco_url":"",
                                     "height":ann_file["imageHeight"],
                                     "width":ann_file["imageWidth"],
                                     "date_captured":"",
                                     "flickr_url":"",
                                     "id":img_idx}

    def add_categories(self):
        keypoints = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
                     "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                     "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                    [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        self.categories = [{"supercategory":"person",
                            "id":1,
                            "name":"person",
                            "keypoints":keypoints,
                            "skeleton":skeleton}]


    def add_annotations(self):
        self.annotations = []
        ann_id = 0
        for img_idx, ann_file in enumerate(self.json_files):
            new_annotation = {}
            shapes = ann_file["shapes"]
            if len(shapes)  == 1 and shapes[0]["shape_type"] == 'circle':
                print("NULL ANNOTATION: {}".format(self.images[img_idx]['file_name']))
                continue
            for shape in shapes:
                group_id = str(shape["group_id"])
                if shape["shape_type"] == 'rectangle':
                    identifier = group_id
                else:
                    identifier = group_id[:-1]
                if(identifier not in new_annotation.keys()):
                    new_annotation[identifier] = {"id":ann_id,
                                                  "iscrowd":0,
                                                  "image_id":img_idx,
                                                  "category_id":1,
                                                  "num_keypoints":0}
                    kpts = OrderedDict()
                    for kpt_name in self.categories[0]["keypoints"]:
                        kpts[kpt_name] = None
                    new_annotation[identifier]["keypoints"] = kpts
                    ann_id += 1
                if shape["shape_type"] == 'rectangle':
                    top_left = shape["points"][0]
                    bottom_right = shape["points"][1]
                    top_left_x = round(top_left[0],2)
                    top_left_y = round(top_left[1],2)
                    bot_right_x = round(bottom_right[0],2)
                    bot_right_y = round(bottom_right[1],2)
                    if bot_right_x < top_left_x:
                      tmp_bot_right = top_left_x
                      top_left_x = bot_right_x
                      bot_right_x = tmp_bot_right
                    if bot_right_y < top_left_y:
                      tmp_bot_right = top_left_y
                      top_left_y = bot_right_y
                      bot_right_y = tmp_bot_right
                    width = round(bot_right_x - top_left_x,2)
                    height = round(bot_right_y - top_left_y,2)

                    new_annotation[identifier]['bbox'] = [top_left_x,top_left_y,
                                                          width,height]
                    new_annotation[identifier]['segmentation'] = [[top_left_x, top_left_y,
                                                                  bot_right_x, top_left_y,
                                                                  bot_right_x, bot_right_y,
                                                                  top_left_x, bot_right_y]]
                    new_annotation[identifier]['area'] = round(width*height,2)
                else:
                    kpt_name = shape["label"]
                    if(len(group_id)<2):
                        print(("Keypoint: {}, of person: {}, from Image: {}, does not specify visibility!"
                              .format(kpt_name,identifier,self.images[img_idx]["file_name"])))
                        visibility = 1
                    else:
                        visibility = int(group_id[-1])
                        if visibility > 2:
                            print(("Visibility of keypoint: {}, of person: {}, from image: {}, does not match standards!"
                              .format(kpt_name, identifier,self.images[img_idx]["file_name"])))
                    kpt_x = round(shape["points"][0][0],2)
                    kpt_y = round(shape["points"][0][1],2) 
                    if(new_annotation[identifier]["keypoints"][kpt_name] is not None):
                        print(("Keypoint: {}, of person: {}, from image: {}, registered twice!"
                              .format(kpt_name,identifier,self.images[img_idx]["file_name"])))
                    new_annotation[identifier]["keypoints"][kpt_name] = [kpt_x,kpt_y,visibility]

                    new_annotation[identifier]["num_keypoints"] += int(visibility>0)

            for i, person_det in new_annotation.items():
                if "bbox" not in person_det.keys():
                    print(("Missing bbox in person: {} from image: {}"
                               .format(i,self.images[img_idx]["file_name"])))
                person_det["keypoints"] = list(person_det["keypoints"].values()) 
                final_kpts = []
                for kpt_idx, kpt in enumerate(person_det["keypoints"]):
                    if kpt==None:
                        print(("Missing keypoint: {} in person: {} of image: {}"
                               .format(self.categories[0]["keypoints"][kpt_idx],i,self.images[img_idx]["file_name"])))
                    else:
                        final_kpts.extend(kpt)
                person_det["keypoints"] = final_kpts
                self.annotations.append(person_det)

    def transform2coco(self,output_dir):
        self.add_info()
        self.add_licenses()
        self.add_images()
        self.add_categories()
        self.add_annotations()
        output_ann_file = {"info":self.info,
                           "licencese":self.licenses,
                           "images":list(self.images.values()),
                           "annotations":self.annotations,
                           "categories":self.categories}
        with open(output_dir + "thermalPose_{}.json".format(args.set), "w") as fp:
            json.dump(output_ann_file, fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform labelme annotations to COCO format")
    parser.add_argument("-i", "--input_path", dest="input_path",
                  help="input path to json annotation files")
    parser.add_argument("-o", "--output_path", dest="output_path",
                  help="path for output annotation file")
    parser.add_argument("-s", "--set", dest="set",
                  help="type of set of images (train, test, etc)")

    args = parser.parse_args()
    dataset = PoseThermal(args.input_path)
    dataset.transform2coco(args.output_path)





