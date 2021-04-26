import cv2
import numpy as np
import os
import glob
import json

#def resize_keypoints():
def resize_dataset(imgs_path, ann_path,output_size, output_dir,visualize = False):
    with open(ann_path,'r') as f:
        data = json.load(f)
    
    output_folder = os.path.join(output_dir,'{}_{}'.format(imgs_path.split("/")[-1],output_size))
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'images'))

    new_anns = []
    anns_img_id = np.array([ann['image_id'] for ann in data['annotations']])
    for img_info in data['images']:
        img_filename = img_info['file_name']
        img_id = img_info['id']
        img = cv2.imread(os.path.join(imgs_path,img_filename))
        height,width = img.shape[:2]
        max_size = width if width > height else height
        if max_size > output_size:
            img = cv2.GaussianBlur(img, (5,5),0)
        if width >= height:
            new_width = output_size
            new_height = int(output_size/width*height)
        else:
            new_width = int(output_size/height*width)
            new_height = output_size
        resized_img = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_LINEAR)
        data['images'][img_id]['height'] = new_height
        data['images'][img_id]['height'] = new_width
        cv2.imwrite(os.path.join(output_folder,img_filename),resized_img)

        ann_ids =np.where(anns_img_id == img_id)[0]
        if len(ann_ids) == 0:
            continue
        anns = [data['annotations'][ann_id] for ann_id in ann_ids]
        for ann in anns:
            kpts = ann['keypoints']
            seg = ann['segmentation']
            bbox = ann['bbox']
            x1 = bbox[0]/width*new_width
            bbox_width = bbox[2]/width*new_width
            y1 = bbox[1]/height*new_height
            bbox_height = bbox[3]/height*new_height
            cv2.rectangle(resized_img,(int(x1),int(y1)),(int(x1+bbox_width),int(y1+bbox_height)),(0,255,0),3)
            xs = kpts[::3]
            ys = kpts[1::3]
            viss = kpts[2::3]
            new_keypoints = []
            for pt_idx in range(len(xs)):
                x = xs[pt_idx]/width*new_width
                y = ys[pt_idx]/height*new_height
                vis = viss[pt_idx]
                new_keypoints = new_keypoints + [x,y,vis]
                cv2.circle(resized_img, (int(x),int(y)), 1, (0,0,255), 2)
            ann['keypoints'] = new_keypoints
            ann['bbox'] = [x1,y1,bbox_width,bbox_height]
            ann['area'] = bbox_width * bbox_height
            ann['segmentation'] = [x1,y1,x1+bbox_width,y1,x1+bbox_width,y1+bbox_height,x1,y1+bbox_height]
            new_anns.append(ann)

        if visualize:
            cv2.imshow('resized_img',resized_img)
            waitKey = cv2.waitKey(0)
            if waitKey & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    data['annotations'] = new_anns
    with open(os.path.join(output_folder,'thermalPose_{}_{}.json'.format(imgs_path.split("/")[-1],output_size)),'w') as f:
        json.dump(data,f)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    imgs_path = '/home/javier/Universidad/memoria/repositorios/CenterNet-Thermal-Human-Pose-Estimation/CenterNet/data/thermal_pose/val'
    ann_path = '/home/javier/Universidad/memoria/repositorios/CenterNet-Thermal-Human-Pose-Estimation/CenterNet/data/thermal_pose/annotations/thermalPose_val.json'
    output_dir = '/home/javier/Universidad/memoria/RESIZED_IMGS'
    resize_dataset(imgs_path, ann_path, 512,output_dir,visualize=False)