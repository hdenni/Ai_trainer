#!/bin/bash

docker run -d -it --rm --name ringfit --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 cwaffles/openpose

docker cp /home/centos/ringfit_web/upload/user.mp4 ringfit:/openpose/examples/media/

docker cp /home/centos/ringfit_web/ringfit_py/run_openpose.sh ringfit:/openpose/

docker exec ringfit /openpose/run_openpose.sh

docker cp ringfit:/openpose/output/ /home/centos/ringfit_web/ringfit_py/

docker stop ringfit

#cp /home/centos/ringfit_web/ForSlave/ringfit_ver01.py /home/centos/ringfit_web/ForSlave/output/

cd /home/centos/ringfit_web/ringfit_py

python3 ringfit_ver03.py

cp /home/centos/ringfit_web/ringfit_py/result_put_fps30.mp4 /home/centos/ringfit_web/static/result/videos/

cp /home/centos/ringfit_web/ringfit_py/result_put_fps15.mp4 /home/centos/ringfit_web/static/result/videos/

cp /home/centos/ringfit_web/ringfit_py/result_put_fps6.mp4 /home/centos/ringfit_web/staic/result/videos/

cp /home/centos/ringfit_web/ringfit_py/ringfit_output_put.json /home/centos/ringfit_web/static/result/json/

cp /home/centos/ringfit_web/ringfit_py/time_graph_put.jpg /home/centos/ringfit_web/static/result/images/

cp /home/centos/ringfit_web/ringfit_py/sim_graph_put.jpg /home/centos/ringfit_web/static/result/images/
