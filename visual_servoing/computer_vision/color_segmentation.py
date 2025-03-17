import cv2
import numpy as np
# import apriltag

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

	### For test cases ###
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	### For continuous streaming ###
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

def cd_color_segmentation(img, template, display=False, line=False):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	### HSV Parameters ###
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Cone #
	hue_val = 18
	hue_win = 15
	sat_val = .50
	sat_win = .10
	val_val = .70
	val_win = .10

	hue_low, hue_high = 2, 22
	saturation_low, saturation_high = 200, 255
	value_low, value_high = 170, 255

	# Line, outside, far #
	# hue_val = 14
	# hue_win = 1.5
	# sat_val = .6
	# sat_win = .1
	# val_val = .6
	# val_win = .20

	# # Line, outside, close #
	# hue_val = 14
	# hue_win = 2
	# sat_val = .6
	# sat_win = .15
	# val_val = .6
	# val_win = .3

	# hue_low, hue_high = hue_val-hue_win, hue_val+hue_win #2 , 30
	# saturation_low, saturation_high = min(sat_val*255-sat_win,0), max(sat_val*255+sat_win,255) # 150, 255
	# value_low, value_high = (val_val-val_win)*255, (val_val+val_win)*255 # 170, 255

	# Filtering Image #
	lower_orange = np.array([hue_low, saturation_low, value_low])
	upper_orange = np.array([hue_high, saturation_high, value_high])
	mask = cv2.inRange(hsv, lower_orange, upper_orange)

	# Processing mask#
	if line:
		erode_kernel = np.ones((4, 4), np.uint8)
		dilate_kernel = np.ones((3, 3), np.uint8)
		mask = cv2.dilate(mask, dilate_kernel, iterations=1)
		mask = cv2.erode(mask, erode_kernel, iterations=1)
	else:
		erode_kernel = np.ones((4, 4), np.uint8)
		dilate_kernel = np.ones((3, 3), np.uint8)
		mask = cv2.erode(mask, erode_kernel, iterations=1)
		mask = cv2.dilate(mask, dilate_kernel, iterations=1)

	# if display is True:
	# 	image_print(mask)

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

CAMERA_MATRIX = np.array([[344.8610534667969, 0, 321.635986328125],
                           [0, 344.8610534667969, 172.66043090820312],
                           [0, 0, 1.0]])
DIST_COEFFS = np.zeros(5)
TAG_SIZE = 0.15

def detect_apriltags(img):
	"""
	Detects AprilTags in an image and calculates their center coordinates and distance to the camera.
	"""
	f_x = CAMERA_MATRIX[0, 0]
	f_y = CAMERA_MATRIX[1, 1]
	c_x = CAMERA_MATRIX[0, 2]
	c_y = CAMERA_MATRIX[1, 2]

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detector = apriltag.Detector()
	detections = detector.detect(img)
	results = []

	for detection in detections:
		tag_id = detection.tag_id
		center = detection.center
		corners = detection.corners

		object_points = np.array([[-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
									[ TAG_SIZE / 2, -TAG_SIZE / 2, 0],
									[ TAG_SIZE / 2,  TAG_SIZE / 2, 0],
									[-TAG_SIZE / 2,  TAG_SIZE / 2, 0]], dtype=np.float32)

		_, rvec, tvec = cv2.solvePnP(object_points, corners, CAMERA_MATRIX, DIST_COEFFS)

		# Convert translation vector (tvec) into x and y distances
		X = tvec[2][0]  # x distance (outward from the camera)
		Y = -tvec[0][0]  # y distance (left/right of the camera)

		results.append({'id': tag_id, 'center': center, 'X': X, 'Y': Y})

		cv2.drawContours(img, [np.int32(corners)], -1, (0, 255, 0), 2)
		cv2.putText(img, f"ID: {tag_id}, X: {X}, Y: {Y}", tuple(np.int32(center)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	image_print(img)
	return results
