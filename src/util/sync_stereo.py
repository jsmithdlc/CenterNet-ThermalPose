import cv2
import glob
import numpy as np

pts_src = {"thermal_9_56_26":[[93,136],[247,130],[479,120],[367,314],[201,357],[318,218],[262,286],\
							  [543,274],[189,183],[336,103],[404,198]],\
		   "thermal_10_25_19":[[398,182],[282,189],[281,279],[408,278],[202,130],[465,189],[91,247],\
		   					   [210,108],[240,259],[538,191],[348,323],[472,248],[276,65],[476,120],\
		   					   [199,359],[328,110]]}
pts_dst = {"webcam_9_56_26":[[30,238],[382,213],[863,139],[636,544],[284,674],[538,372],[437,508],\
							 [957,472],[283,323],[545,148],[694,318]], \
		   "webcam_10_25_19":[[610,311],[396,332],[413,498],[662,471],[286,219],[824,287],[50,474],\
		   					 [296,176],[383,461],[1012,264],[553,573],[845,403],[411,77],[851,140],\
		   					 [285,678],[466,182]]}

im_src = cv2.imread('../../stereo_captures/thermal/thermal_10_25_19.jpg')
im_dst = cv2.imread('../../stereo_captures/webcam_undistorted/webcam_10_25_19.jpg')


im_kpts_src = im_src.copy()
im_kpts_dst = im_dst.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0,255,0)
lineType = 2
for i,pt_src in enumerate(pts_src["thermal_10_25_19"]):
	pt_dst = pts_dst["webcam_10_25_19"][i]
	im_kpts_src = cv2.circle(im_kpts_src, (pt_src[0],pt_src[1]), 10, (0,0,255),1)
	im_kpts_dst = cv2.circle(im_kpts_dst, (pt_dst[0],pt_dst[1]), 10, (0,0,255),1)
	im_kpts_src = cv2.putText(im_kpts_src,str(i),(pt_src[0],pt_src[1]), font, fontScale,fontColor,lineType)
	im_kpts_dst = cv2.putText(im_kpts_dst,str(i),(pt_dst[0],pt_dst[1]), font, fontScale,fontColor,lineType)


cv2.imshow("Keypoints Source", im_kpts_src)
cv2.imshow("Keypoints Destination", im_kpts_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst_obj = np.array(pts_src["thermal_10_25_19"])
src_obj = np.array(pts_dst["webcam_10_25_19"])
"""
#h1, status1 = cv2.findHomography(np.array(pts_src["thermal_9_56_26"]), np.array(pts_dst["webcam_9_56_26"]),confidence=0.999)
#h2, status2 = cv2.findHomography(np.array(pts_src["thermal_10_25_19"]), np.array(pts_dst["webcam_10_25_19"]))
"""
h1, status1 = cv2.findHomography(np.array(pts_dst["webcam_10_25_19"]),np.array(pts_src["thermal_10_25_19"]),confidence =0.9999)
#h2, status2 = cv2.findHomography(np.array(pts_dst["webcam_10_25_19"]),np.array(pts_src["thermal_10_25_19"]))
idx_affine = [13,7,14]
#h1= cv2.getAffineTransform(np.array([src_obj[idx_affine[0]],src_obj[idx_affine[1]],src_obj[idx_affine[2]]],np.float32),\
#	np.array([dst_obj[idx_affine[0]],dst_obj[idx_affine[1]],dst_obj[idx_affine[2]]],np.float32))
#h = (h1+h2)/2.0
h = h1
thermal_paths = glob.glob('../../stereo_captures/thermal/*.jpg')
rgb_paths = ['../../stereo_captures/webcam_undistorted/webcam_' + "_".join(img_path.split("/")[-1].split("_")[1:]) for img_path in thermal_paths]
for im_idx, im_path in enumerate(thermal_paths):
	im_thermal = im_src
	im_rgb = im_dst
	#im_thermal = cv2.imread(im_path)
	#im_rgb = cv2.imread(rgb_paths[im_idx])
	#im_out = cv2.warpPerspective(im_thermal, h, (im_rgb.shape[1],im_rgb.shape[0]))
	im_out = cv2.warpPerspective(im_rgb, h, (im_thermal.shape[1],im_thermal.shape[0]))

	#im_out = cv2.warpAffine(im_thermal, h, (im_rgb.shape[1],im_rgb.shape[0]))
	#im_out = cv2.warpAffine(im_rgb, h, (im_thermal.shape[1],im_thermal.shape[0]))

	# Display images

	#cv2.imshow("Source Image", im_src)

	#cv2.imshow("Destination Image", im_dst)

	#cv2.imshow("Warped Source Image", im_out)

	alpha = 0.5
	beta = 1.0 - alpha
	weighted_img = cv2.addWeighted(im_out, alpha, im_thermal, beta, 0.0)
	cv2.imshow('Weighted', weighted_img)
	cv2.waitKey(0)

