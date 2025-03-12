import cv2
import numpy as np
# from cv_test import test_algorithm

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

def cd_color_segmentation(img, template, display=False):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# HSV Parameters #
	hue_low, hue_high = 2, 30
	saturation_low, saturation_high = 150, 255
	value_low, value_high = 170, 255

	# Filtering Image #
	lower_orange = np.array([hue_low, saturation_low, value_low])
	upper_orange = np.array([hue_high, saturation_high, value_high])
	mask = cv2.inRange(hsv, lower_orange, upper_orange)

	# Processing mask with closing #
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.dilate(mask, kernel, iterations=1)
	mask = cv2.erode(mask, kernel, iterations=1)

	# Drawing box around biggest contour #
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	bounding_box = ((0, 0), (0, 0))
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.circle(img, (int(x + w/2), y + h), 1, (255, 0, 0), -1)
		bounding_box = ((x, y), (x + w, y + h))

	if display is True:
		image_print(img)

	return bounding_box
