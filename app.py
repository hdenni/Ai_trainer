# -*- coding:utf-8 -*-
#!usr/bin env python

import cv2
import json
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit

# emurated camera
from webcamvideostream import WebcamVideoStream

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/webcam', methods=['POST', 'GET'])
def webcam():
	return render_template("webcam2.html") # webcam.html은 작동함 2는 실험중

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
#	for video_frame in generate(WebcamVideoStream()):
#		socketio.emit('from_flask', {'data':video_frame}, namespace='/test')
	return Response(generate(WebcamVideoStream()),
					mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/save_record', methods=["POST", "GET"])
def save_record():
	if request.method == 'POST':
		f = request.files['file']
		filename = str(request.files['title'])
		f.save('upload/'+filename)
		return 'success'

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
	
	# docker openpose와 관련된 shell파일 실행
	# result와 관련된 변수 정의
	# return render_tempate("result.html", ...)
	return "Success"

@app.route('/comment')
def comment():
	result_json="resource/result_002.json"

	with open(result_json, 'r') as json_file:
		data=json.load(json_file)

	return render_template("comment.html", data=data)

@app.route('/result')
def result():
	# To-Do: 스켈레톤 이미지 띄워보기
	# 아 스켈레톤 그대로 할지 openpose 쓸지 결정해야함!! -> 그려줄게
	result_json = "resource/result_002.json"
	sex_json = "upload/28-1_001-C09_mpi.json"
	
	with open(result_json, 'r') as json_file:
		data=json.load(json_file)

	with open(sex_json, "r") as json_file:
		sex=json.load(json_file)

	return render_template('result2.html', data=data, sex=sex)

if __name__=='__main__':
	app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
