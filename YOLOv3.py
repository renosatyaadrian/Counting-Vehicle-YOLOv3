# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

classname = []
list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
def get_vehicle_count(boxes, class_names):
	total_vehicle_count = 0 # total vechiles present in the image
	dict_vehicle_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		class_name = class_names[i]
		# print(i,".",class_name)
		if(class_name in list_of_vehicles):
			total_vehicle_count += 1
			dict_vehicle_count[class_name] = dict_vehicle_count.get(class_name,0) + 1

	return total_vehicle_count, dict_vehicle_count

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3-transf.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = prop
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream

list_of_vehicles = ["car","bus","motorbike","truck","bicycle"]

# penghitung mobil
penghitung = 0

# start
upper_left = (425, 0)
bottom_right = (853, 480)

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	#frame = rect_img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	# cv2.line(frame, (200, 455), (640,455), (0,127,255), 3) plengkung gading
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				#if 20 < x < 600 and y == 340 :
				#	penghitung+=1
				#	cv2.line(frame, (20, 350), (600, 350), (0,0,255), 3)



				# update our list of bounding box coordinates,
				# confidences, and class IDs
				if x >200:
					boxes.append([x, y, int(width), int(height)])
					#print(confidence)
					confidences.append(float(confidence))
					#print(classIDs)
					classIDs.append(classID)
					classname.append(LABELS[classID])

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	#print(boxes)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	#print(idxs)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			
			if LABELS[classIDs[i]] != "person":
				
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				
				
				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				
				#if x > 200:
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				#cv2.putText(frame, "kendaraan: "+str(penghitung), (425, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
				#cv2.line(frame, (20, 455), (640,455), (0,127,255), 3)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	
	total_vehicles, each_vehicle = get_vehicle_count(boxes, classname)
	print("Total vehicles in image", total_vehicles)
	print("Each vehicles count in image", each_vehicle)

	cv2.putText(frame, "Total Kendaraan: "+str(total_vehicles), (0, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123, 0, 255),2)
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk

	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
