# -*- coding:utf-8 -*-
# import the necessary packages
from threading import Thread
import cv2

class WebcamVideoStream:
		def __init__(self, src=0):
				# initialize the video camera stream and read the first frame
				# from the stream
				print("init")
				self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

		def __del__(self):
				self.video.release()

		def get_frame(self):
				ret, frame = self.video.read()
				if ret:
						_, jpeg = cv2.imencode('.jpg', frame)
						return jpeg.tobytes()
				else:
						return None
