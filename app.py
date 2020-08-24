# -*- coding:utf-8 -*-
#!usr/bin env python

import cv2
import json
from flask import Flask, render_template, Response, request, url_for, redirect
import os

# emurated camera
from webcamvideostream import WebcamVideoStream

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/test')
def test():
	return render_template("webcam3.html")

@app.route('/webcam')
def webcam():
	# webcam.html: webcam streaming
	# webcam2.html: webcam streaming + send video to server
	# webcam3.html: by python opencv + flask
	return render_template("webcam2.html") 

def generate(camera):
	#out = cv2.VideoWriter('save.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25.0, (1280, 720))
	while True:
		frame = camera.get_frame()

		if frame:
			yield (b'--frame\r\n'
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

# @app.route('/record_status', methods=['POST'])
# def record_status():
# 	json = request.get_json()
# 	status = json['status']
# 	if status == "true":
# 		WebCamCamerai().

@app.route('/upload')
def render_file():
	return render_template('upload.html')

@app.route('/uploadFile', methods=["POST"])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		#f.save('upload/'+f.filename)
		#f.save('upload/'+'user_000.mp4')
		f.save('static/result/videos/user_000.mp4')
	
	openpose()
	return redirect(url_for('result'))

# openpose 관련 shell파일 실행
def openpose():
	os.system('/home/centos/ringfit_web/ringfit_ML/main.sh')

@app.route('/result')
def result():
	result_json="static/result/json/result.json"

	with open(result_json, 'r') as json_file:
		data=json.load(json_file)

	# value = list(map(float, data.values()[:5]))

	value = list()
	for i in ["1", "2", "3", "4", "5"]:
		value.append(float(data[i]))

	if value[0] < 0: data["1"] = "평균보다 무릎이 약 " + str(-1 * value[0]) + "도 덜 구부러졌습니다."
	else: data["1"] = "평균보다 무릎이 약 " + str(value[0]) + "도 더 구부러졌습니다."

	if value[1] < 0: data["2"] = "평균보다 허리가 약 " + str(-1 * value[1]) + "도 덜 굽혀졌습니다."
	else: data["2"] = "평균보다 허리가 약 " + str(value[1]) + "도 더 굽혀졌습니다."

	if value[2] < 0: data["3"] = "평균보다 약 " + str(-1 * value[2]) + " 빠르게 앉았습니다."
	else: data["3"] = "평균보다 약 " + str(value[2]) + " 느리게 앉았습니다."

	if value[3] < 0: data["4"] = "평균보다 약 " + str(-1 * value[3]) + " 빠르게 일어났습니다."
	else: data["4"] = "평균보다 약 " + str(value[3]) + " 느리게 일어났습니다."

	data["5"] = "운동의 전체적인 유사도는 " + str(value[4]) + "% 입니다."

	return render_template('result.html', data=data)

if __name__=='__main__':
	app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
