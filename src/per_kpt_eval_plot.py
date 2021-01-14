import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
src_img = "./human_figure.png"
scores_hg_best = [0.537,0.523,0.524,0.673,0.690,0.888,0.890,0.729,0.788,0.528,0.578,0.865,0.886,0.744,0.789,0.706,0.735]
scores_hg_paper = [0.387,0.401,0.373,0.426,0.425,0.659,0.683,0.546,0.581,0.359,0.397,0.659,0.652,0.560,0.589,0.551,0.534]
scores_dla_best = [0.550,0.551,0.546,0.637,0.697,0.849,0.858,0.691,0.735,0.480,0.520,0.846,0.862,0.743,0.765,0.661,0.659]
scores_dla_paper = [0.376,0.409,0.383,0.418,0.380,0.633,0.644,0.479,0.511,0.284,0.350,0.654,0.635,0.531,0.567,0.481,0.484]


def plot_kpt_eval(scores,backbone, name):
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
	ax.set_title('{} [{}]\nAP por keypoint'.format(backbone,name))
	plt.savefig("../figures/exp_plots/per_kpt/{}_{}_kpt_eval.pdf".format(backbone,name))
	plt.show()

if __name__ == '__main__':
	plot_kpt_eval(scores_dla_paper,"DLA","paper")
	plot_kpt_eval(scores_dla_best,"DLA","mejor")
	plot_kpt_eval(scores_hg_paper,"Hourglass","paper")
	plot_kpt_eval(scores_hg_best,"Hourglass","mejor")

