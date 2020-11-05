import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
src_img = "./human_figure.png"
scores = [0.373,0.426,0.371,0.459,0.491,0.706,0.740,0.561,0.590,0.323,0.369,0.725,0.721,0.626,0.646,0.590,0.603]
kpt_names = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',\
             'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee',\
             'left_ankle','right_ankle']
kpt_coord = [(229,95),(243,72),(216,72),(263,91),(193,91),(315,183),(144,183),(324,324),\
			 (134,324),(355,438),(101,438),(289,423),(163,423),(252,594),(193,590),(240,772),\
			 (200,771)]

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
plt.show()
"""
cv2.imshow("imagen",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""