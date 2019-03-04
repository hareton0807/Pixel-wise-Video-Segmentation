# USAGE
# python superpixel.py --image raptors.png

# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import time

def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0
 
	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
 
		# increment the total number of frames read
		total += 1
 
	# return the total number of frames in the video file
	return total



# load the video and divide it into frames
vidcap = cv2.VideoCapture("test2.mp4")
print("Frames: ",count_frames_manual(vidcap))
vidcap = cv2.VideoCapture("test2.mp4")
success,image = vidcap.read()

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('myOutput.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (int(vidcap.get(3)),int(vidcap.get(4))))
count = 0
while True:
        
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        ret,image = vidcap.read()
        if count == 0:
                t1 = time.time()
        if ret == True:
                # load the image and convert it to a floating point data type
                image = img_as_float(image)
                count += 1
                if count % 10 == 0:
                        print(count)

                # apply SLIC and extract (approximately) the supplied number
                # of segments
                segments = slic(image, n_segments = 100, sigma = 5)
                image = mark_boundaries(image,segments)
                image = img_as_ubyte(image)
                out.write(image)
                if count == 1:
                        print(time.time()-t1)
        else:
                break

        
vidcap.release()
out.release()
cv2.destroyAllWindows() 
print("Done!")

                
        
