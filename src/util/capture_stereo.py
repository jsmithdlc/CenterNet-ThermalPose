import cv2
import numpy as np
import os
from datetime import datetime
import argparse
import time
def main(opts):
	# Link to thermal camera
	ret_thermal = True
	link_color = "rtsp://10.0.0.12/h264.sdp?res=half&x0=0&y0=0&x1=2560&y1=1920&qp=26&fps=2&doublescan=0&ssn=61552"
	link_thermal = "rtsp://10.0.0.11:554/ch0"
	# Initialize webcam + thermal cameras video captures
	time.sleep(opts.delay_seconds)
	os.system('play -nq -t alsa synth {} sine {}'.format(1,200))
	cap_webcam = cv2.VideoCapture(link_color)
	#cap_thermal = cv2.VideoCapture(link_thermal)

	# Creates directories for outputs, if needed
	if not os.path.isdir(opts.output_dir + "/webcam"):
		os.mkdir(opts.output_dir+"/webcam")
		os.mkdir(opts.output_dir+"/thermal")

	new_height = opts.height # Height for new resized frames from videocaptures
	# Reads the first frame from each camera and calculates new dimensions
	ret_webcam, webcam_frame = cap_webcam.read()
	#ret_thermal, thermal_frame = cap_thermal.read() 
	if not ret_thermal or not ret_webcam:
		print("Didn't detect initial frames")
		return
	ar_webcam = webcam_frame.shape[0]/webcam_frame.shape[1]    # webcam frames aspect ratio
	#ar_thermal = thermal_frame.shape[0]/thermal_frame.shape[1] # thermal frames aspect ratio
	dim_webcam = (int(new_height/ar_webcam),new_height)
	#dim_thermal = (int(new_height/ar_thermal),new_height)

	#start_time = time.time()
	delay = opts.delay_seconds * 1000

	n_images = 0
	duration = 1
	freq = 440
	#cv2.waitKey(delay)
	#time.sleep(opts.delay_seconds)
	# Shows images until user presses "q"
	#while(True):
		# Grabs the next frame from each capturing device, several times
	cap_webcam.grab()
	#cap_thermal.grab()
	"""
	for i in range(1):
	    cap_webcam.grab()
	    cap_thermal.grab()
	"""
	# Retrieves next frame from each capturing device 
	ret_webcam, webcam_frame = cap_webcam.retrieve()
	#ret_thermal, thermal_frame = cap_thermal.retrieve()  
	#cv2.waitKey(5000)
	# Display images only if both frames were successfully captured
	if ret_webcam and ret_thermal:
		img_webcam = cv2.resize(webcam_frame,dim_webcam,cv2.INTER_AREA)
		#img_thermal = cv2.resize(thermal_frame,dim_thermal,cv2.INTER_AREA)
		# Merges both frames for visualization purposes
		#merged = np.concatenate((img_webcam, img_thermal), axis=1) 
		#cv2.imshow("RGB - THERMAL",merged)
		 # If user presses "space", captures both frames

		 # For storing resized images, instead of original ones
		#if opts.store_resized:
		#	webcam_frame = img_webcam
		#	thermal_frame = img_thermal
		# Retrieves actual time
		now = datetime.now()
		if cv2.waitKey(33) == 32:
			# Writes images to output directories
			cv2.imwrite(opts.output_dir + "/webcam/webcam_{}_{}_{}.jpg".format(now.hour,now.minute,
				now.second),webcam_frame)
			#cv2.imwrite(opts.output_dir + "/thermal/thermal_{}_{}_{}.jpg".format(now.hour,now.minute,
			#	now.second),thermal_frame)
		elif delay >= 0:
			cv2.imwrite(opts.output_dir + "/webcam/webcam_{}_{}_{}.jpg".format(now.hour,now.minute,
				now.second),webcam_frame)
			#cv2.imwrite(opts.output_dir + "/thermal/thermal_{}_{}_{}.jpg".format(now.hour,now.minute,
			#	now.second),thermal_frame)
			n_images +=1

				#start_time = time.time()
	#cv2.waitKey(delay)
	"""
	if n_images == 20:
		break

	# If user presses "q", exits program
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	"""

	"""
	# If user presses "space", captures both frames
	elif cv2.waitKey(33) == 32:
		# Writes images to output directories
		cv2.imwrite(opts.output_dir + "/webcam/webcam_{}_{}_{}.jpg".format(now.hour,now.minute,
			now.second),webcam_frame)
		cv2.imwrite(opts.output_dir + "/thermal/thermal_{}_{}_{}.jpg".format(now.hour,now.minute,
			now.second),thermal_frame)
	elif delay >= 0:
		cv2.imwrite(opts.output_dir + "/webcam/webcam_{}_{}_{}.jpg".format(now.hour,now.minute,
			now.second),webcam_frame)
		cv2.imwrite(opts.output_dir + "/thermal/thermal_{}_{}_{}.jpg".format(now.hour,now.minute,
			now.second),thermal_frame)
		cv2.waitKey(delay)
	"""

	# When everything done, release the capture
	cap_webcam.release()
	#cap_thermal.release()
	cv2.destroyAllWindows()
	os.system('play -nq -t alsa synth {} sine {}'.format(duration,freq))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Capture stereo pair images')
	parser.add_argument('--output_dir', type=str,
                    	help='output directory for image pairs')
	parser.add_argument('--height', type = int,default=300,
	                	help='new_height for displaying image pair')
	parser.add_argument('--store_resized', action='store_true',
	                	help='wether to store pair with resized sizes')
	parser.add_argument('--delay_seconds', type=int, default = -1,
	                	help='delay for automatic capture')
	args = parser.parse_args()
	main(args)

