"""To execute the program on Raspberry Pi : python3 faceRecognitionRPi.py"""
# Written by Shubham Das
# Reffered from https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/ 

"""Please go through the above link to understand the concept in depth. He has seperately encodded the 
images and then recognised them. But I combined both the programs to one single program and removed 
argument parser from it. Please feel free to contact me for any issues. """ 

# import all the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import time
import cv2
from imutils import paths
import os

# load the known faces and embeddings along with OpenCV's Haar
# open cv haarcascade for face detection
data_folder="dataset/"
imagePaths = list(paths.list_images(data_folder))
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hog=cv2.HOGDescriptor()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# FPS counter starting
fps = FPS().start()
# declare a list to store the known encodings of the face and the names of the face.
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	# now we have to extract the name of the person from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	Load_name = imagePath.split(os.path.sep)[-2]

	# load the image and we need to convert it from openCV ordering (BGR) to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
	Load_boxes = face_recognition.face_locations(rgb_image, model=hog)

 
	# now computation of the facial embedding for the face
	Load_encodings = face_recognition.face_encodings(rgb_image, Load_boxes)
	for Load_encoding in Load_encodings:
		# add each encoding + name to our set of known names and encodings
		knownEncodings.append(Load_encoding)
		knownNames.append(Load_name)
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	""" convert the input frame from (1) BGR to grayscale (for face detection)
     and (2) from BGR to RGB (for face recognition)"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE)

	"""OpenCV returns bounding box coordinates in (x, y, w, h) order but we need 
    them in (top, right, bottom, left) order, so we need to do a bit of reordering"""
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# trying to match each face in the input image to our known encodings (knownEncodings)
		matches = face_recognition.compare_faces(knownEncodings, encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			""" find the indexes of all matched faces then initialize a dictionary to count the total 
            number of times each face was matched"""
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for each recognized face face
			for i in matchedIdxs:
				name =knownNames[i]
				counts[name] = counts.get(name, 0) + 1

			"""determine the recognized face with the largest number of votes 
            (note: in the event of an unlikely tie Python will select first entry in the dictionary)"""
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()