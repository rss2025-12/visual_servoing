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
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

def cd_color_segmentation(img, template, display=False, paper=False):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	if paper is True:
		img = detect_paper_corners(img)
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

def detect_paper_corners(image):
	height, width = image.shape[:2]
	cropped_image = image[height//2:, :]

	hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
	lower_white = np.array([0, 0, 150])
	upper_white = np.array([180, 20, 190])
	mask = cv2.inRange(hsv, lower_white, upper_white)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if contours:
		contour = max(contours, key=cv2.contourArea)
		epsilon = 0.02 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)

		# TODO: Draw and print top left, top right, bottom left, bottom right pixel
		approx = sorted(approx, key=lambda x: x[0][1])  # Sort by y-coordinate

		if len(approx) >= 4:
			# Separate the points into top and bottom
			top_points = sorted(approx[:len(approx)//2], key=lambda x: x[0][0])  # Sort top points by x-coordinate
			bottom_points = sorted(approx[len(approx)//2:], key=lambda x: x[0][0])  # Sort bottom points by x-coordinate

			# Identify top-left, top-right, bottom-left, bottom-right
			top_left = top_points[0]
			top_right = top_points[-1]
			bottom_left = bottom_points[0]
			bottom_right = bottom_points[-1]

			top_left[0][1] += height // 2
			top_right[0][1] += height // 2
			bottom_left[0][1] += height // 2
			bottom_right[0][1] += height // 2

			print("Top-left:", tuple(top_left[0]))
			print("Top-right:", tuple(top_right[0]))
			print("Bottom-left:", tuple(bottom_left[0]))
			print("Bottom-right:", tuple(bottom_right[0]))

			# Draw the corners on the image
			cv2.circle(image, tuple(top_left[0]), 5, (0, 0, 255), -1)  # Red dot for top-left
			cv2.circle(image, tuple(top_right[0]), 5, (0, 0, 255), -1)  # Red dot for top-right
			cv2.circle(image, tuple(bottom_left[0]), 5, (0, 0, 255), -1)  # Red dot for bottom-left
			cv2.circle(image, tuple(bottom_right[0]), 5, (0, 0, 255), -1)  # Red dot for bottom-right

		return image

	return image