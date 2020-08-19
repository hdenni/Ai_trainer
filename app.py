# -*- coding:utf-8 -*-
#!usr/bin env python

import cv2
import json
from flask import Flask, render_template, Response, request

# emurated camera
# from webcamvideostream import WebcamVideoStream

app = Flask(__name__)


@app.route('/')
def home():
	return render_template("home.html")

@app.route('/webcam')
def webcam():
	return render_template("webcam2.html") # webcam.html은 작동함 2는 실험중

@app.route('/upload')
def render_file():
	return render_template('upload.html')

@app.route('/uploadFile', methods=["POST"])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		filename = 'user_video.mp4'
		f.save('upload/'+filename)
	return 'success'

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
