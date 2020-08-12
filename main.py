# -*- coding:utf-8 -*-
#!usr/bin env python

import cv2
import json
from flask import Flask, render_template, Response, request

# emurated camera
from webcamvideostream import WebcamVideoStream

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/webcam')
def webcam():
	return render_template("webcam.html")

def generate(camera):
	while True:
		frame = camera.get_frame()
		if frame:
			yield(b'--frame\r\n'
				  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		else:
			print("No Frame")
			pass

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
	return Response(generate(WebcamVideoStream()),
					mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/upload')
def render_file():
	return render_template('upload.html')

@app.route("/fileUpload", methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save('upload/'+f.filename)

		s = request.form['radio_sex']
		json_data = {"sex":s}

		with open("upload/"+f.filename[:-4]+".json", "w") as json_file:
			json.dump(json_data, json_file)

	return "Success"

@app.route('/success')
def success():
	return render_template("success.html")

@app.route('/result')
def result():
	# To-Do: 스켈레톤 이미지 띄워보기
	# 아 스켈레톤 그대로 할지 openpose 쓸지 결정해야함!!
	result_json = "resource/result_002.json"
	sex_json = "upload/28-1_001-C09_mpi.json"
	
	with open(result_json, 'r') as json_file:
		data=json.load(json_file)

	with open(sex_json, "r") as json_file:
		sex=json.load(json_file)

	return render_template('result.html', data=data, sex=sex)

if __name__=='__main__':
	app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
