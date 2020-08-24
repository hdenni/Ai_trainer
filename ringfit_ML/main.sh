#!/bin/bash

docker run -d -it --rm --name ringfit --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 cwaffles/openpose

docker cp /home/centos/ringfit_web/static/result/videos/user_000.mp4 ringfit:/openpose/examples/media/

docker cp /home/centos/ringfit_web/ringfit_ML/run_openpose.sh ringfit:/openpose/

docker exec ringfit /openpose/run_openpose.sh

docker cp ringfit:/openpose/output /home/centos/ringfit_web/ringfit_ML/

cp -r /home/centos/ringfit_web/ringfit_ML/output /home/centos/ringfit_web/ringfit_ML/000

#docker stop ringfit

#cp /home/centos/ringfit_web/ForSlave/ringfit_ver01.py /home/centos/ringfit_web/ForSlave/output/

cd /home/centos/ringfit_web/ringfit_ML

python3 ringfit_ML_ver02.py

cp /home/centos/ringfit_web/ringfit_ML/result_000_fps30.webm /home/centos/ringfit_web/static/result/videos/result.webm

#cp /home/centos/ringfit_web/ringfit_ML/result_put_fps15.mp4 /home/centos/ringfit_web/static/result/videos/

#cp /home/centos/ringfit_web/ringfit_ML/result_put_fps6.mp4 /home/centos/ringfit_web/staic/result/videos/

cp /home/centos/ringfit_web/ringfit_ML/ringfit_output_000.json /home/centos/ringfit_web/static/result/json/result.json

cp /home/centos/ringfit_web/ringfit_ML/time_graph_000.jpg /home/centos/ringfit_web/static/result/images/graph.jpg

cp /home/centos/ringfit_web/ringfit_ML/sim_graph_000.jpg /home/centos/ringfit_web/static/result/images/data1.jpg

cp /home/centos/ringfit_web/ringfit_ML/sim_graph_000.jpg /home/centos/ringfit_web/static/result/images/data2.jpg
