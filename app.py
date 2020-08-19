# -*- coding:utf-8 -*-
#!usr/bin env python

import cv2
import json
from flask import Flask, render_template, Response, request, url_for, redirect

# emurated camera
# from webcamvideostream import WebcamVideoStream

app = Flask(__name__)


@app.route('/')
def home():
	return render_template("home.html")

@app.route('/webcam')
def webcam():
	# webcam.html: webcam streaming
	# webcam2.html: webcam streaming + send video to server
	# webcam3.html: by python opencv + flask
	return render_template("webcam2.html") 

@app.route('/upload')
def render_file():
	return render_template('upload.html')

@app.route('/uploadFile', methods=["POST"])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		filename = 'user_video.webm'
		f.save('upload/'+filename)
	
	# docker 실행
	# result.html로 넘어가도록
	return redirect(url_for('result'))

@app.route('/comment')
def comment():
	result_json="resource/result_002.json"

	with open(result_json, 'r') as json_file:
		data=json.load(json_file)

	value = list(map(float, data.values()))

	if value[0] < 0: data["1"] = "평균보다 무릎이 약 " + int(value[0]) + "도 덜 구부러졌습니다."
	else: data["1"] = "평균보다 무릎이 약 " + str(value[0]) + "도 더 구부러졌습니다."

	if value[1] < 0: data["2"] = "평균보다 허리가 약 " + str(value[1]) + "도 덜 굽혀졌습니다."
	else: data["2"] = "평균보다 허리가 약 " + str(value[1]) + "도 더 굽혀졌습니다."

	if value[2] < 0: data["3"] = "평균보다 약 " + str(value[2]) + " 빠르게 앉았습니다."
	else: data["3"] = "평균보다 약 " + str(value[2]) + " 느리게 앉았습니다."

	if value[3] < 0: data["4"] = "평균보다 약 " + str(value[3]) + " 빠르게 일어났습니다."
	else: data["4"] = "평균보다 약 " + str(value[3]) + " 느리게 일어났습니다."

	data["5"] = "운동의 전체적인 유사도는 " + str(value[4]) + "% 입니다."

	return render_template("comment.html", data=data)

@app.route('/result')
def result():	
	return render_template('result.html')

if __name__=='__main__':
	app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
