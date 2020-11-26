import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
src_img = "./human_figure.png"
scores_hg_paper = [0.395,0.384,0.375,0.392,0.414,0.637,0.650,0.538,0.544,0.350,0.386,0.648,0.659,0.540,0.582,0.522,0.538]
scores_dla_paper = [0.381,0.410,0.382,0.427,0.376,0.626,0.639,0.485,0.507,0.284,0.339,0.648,0.634,0.530,0.572,0.484,0.495]
scores_best_dla = [0.557,0.555,0.546,0.642,0.708,0.855,0.864,0.696,0.739,0.483,0.521,0.846,0.866,0.755,0.774,0.679,0.673]
scores_best_hg = [0.546,0.528,0.524,0.672,0.693,0.886,0.892,0.731,0.788,0.524,0.574,0.863,0.887,0.751,0.790,0.709,0.736]


kpt_names = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',\
             'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee',\
             'left_ankle','right_ankle']
kpt_coord = [(229,95),(243,72),(216,72),(263,91),(193,91),(315,183),(144,183),(324,324),\
			 (134,324),(355,438),(101,438),(289,423),(163,423),(252,594),(193,590),(240,772),\
			 (200,771)]


scores = scores_hg_paper
kpt_map = {name:{'xy':kpt_coord[i],'score':scores[i]} for i,name in enumerate(kpt_names)}
img = cv2.imread(src_img)

cmap = matplotlib.cm.get_cmap('hot')
for kpt in kpt_names:
	score = kpt_map[kpt]['score']
	xy = kpt_map[kpt]['xy']
	rgb = (np.array(cmap(score))*255)[:-1].astype(np.int)
	rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]))
	#radius = int(score*20)
	radius = 12
	cv2.circle(img,xy,radius,rgb,thickness=-1,lineType=cv2.LINE_AA)
	#cv2.putText(img,str(score),xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

fig, ax = plt.subplots()
im = ax.imshow(img,cmap=cmap)
ax.axis('off')
cax = fig.add_axes([0.65, 0.15, 0.03, 0.7])
cbar_ticks = np.array([0,25,50,75,100])
cbar = fig.colorbar(im, cax=cax, orientation='vertical',ticks=((cbar_ticks*255)/100).astype(int))

cbar.ax.set_yticklabels(["{}%".format(value) for value in cbar_ticks]) 
ax.set_title('Model Evaluation at Keypoint Level (AP)')
output_path = "../figures/per_kpt/"
plt.savefig(output_path + 'hg_3x_paper')
plt.show()
"""
cv2.imshow("imagen",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""