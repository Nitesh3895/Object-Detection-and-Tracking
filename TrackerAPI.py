from DetectAPI import DetectAPI
import cv2
import dlib
import numpy as np
import time
import multiprocessing
import pandas as pd

def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = inputQueue.get()

		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the label + bounding box coordinates to the output
			# queue
			outputQueue.put((label, (startX, startY, endX, endY)))

windowsName = "output"

#Input Video
cam = cv2.VideoCapture('/home/nitesh/self/03dec/test3.mp4')
fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)
cv2.namedWindow(windowsName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowsName, 640, 480)

#Output Video:
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps, (640,480))

#Create Object for Detection
OBJECT_DETECTAPI = DetectAPI()


#Profiling:
start_time = time.time()

label = 3 # car class label
skipFrames = 30
frameCounter = 0
nobbox = 0

# initialize our list of queues -- both input queue and output queue
# for *every* object that we will be tracking
inputQueues = []
outputQueues = []
processes = []
STATUS = ''

# create two new input and output queues,
iq = multiprocessing.Queue()
oq = multiprocessing.Queue()

to_write = [[0], [0], [0], [0], [0]]
#Looping over all the frames in the video:

while(cam.isOpened()):
	
	ret, frame = cam.read()
	frameCounter += 1
	if ret == False:
		break

	frame = cv2.resize(frame, (640, 480))
	if frameCounter == 0:
		BBOXES, CENTERS, OBJECTS = OBJECT_DETECTAPI.get_object(frame)
		for box in BBOXES:
			box = (box[0], box[2], box[1], box[3])
			to_write[0].append(frameCounter/fps)
			for i, j in enumerate(box):
				to_write[i+1].append(j)


	if frameCounter % skipFrames == 0:
		BBOXES, CENTERS, OBJECTS = OBJECT_DETECTAPI.get_object(frame)
		
		if BBOXES == None:
			STATUS = 'NO_DETECTING'
		else:
			global nobbox
			nobbox = len(BBOXES)
			BBOXES = [[int(i) for i in j] for j in BBOXES]
			STATUS = 'DETECTION====>>>>>'

			for box in BBOXES:
				cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2, 1)
				box = (box[0], box[2], box[1], box[3])

				to_write[0].append(frameCounter/fps)
				for i, j in enumerate(box):
					to_write[i+1].append(j)
				print('Detection box', box)
				inputQueues.append(iq)
				outputQueues.append(oq)

				# spawn a daemon process for a new object tracker
				p = multiprocessing.Process(
					target=start_tracker,
					args=(box, label, frame, iq, oq))
				p.daemon = True
				processes.append(p)
				p.start()
	else:
		STATUS = 'TRACKING--->>'
		# loop over each of our input ques and add the input RGB
		# frame to it, enabling us to update each of the respective
		# object trackers running in separate processes
		for iq in inputQueues:
			iq.put(frame)
		dummy = []
		# loop over each of the output queues
		for oq in outputQueues:
			# grab the updated bounding box coordinates for the
			# object -- the .get method is a blocking operation so
			# this will pause our execution until the respective
			# process finishes the tracking update
			(label, (startX, startY, endX, endY)) = oq.get()
			box = (startX, endX, startY, endY)
			dummy.append(box)

		for box in dummy[-nobbox:]:
			print('Tracking box----->', box)
			cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 255, 255), 2, 1)
			to_write[0].append(frameCounter/fps)
			for i, j in enumerate(box):
				to_write[i+1].append(j)

	print('::::: STATUS :::::', STATUS)
	out.write(frame)

	cv2.imshow(windowsName, frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):

		break

data = {'secs':to_write[0], 'x1':to_write[1], 'y1':to_write[2], 'x2':to_write[3], 'y2':to_write[4]}
df = pd.DataFrame(data)
df.to_csv('out.csv')

# #Completing the processes:
# for p in processes:
# 	p.join()

end_time = time.time()
print('Time Taken:', str(end_time - start_time))