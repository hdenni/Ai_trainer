#!/bin/bash

mkdir output

./build/examples/openpose/openpose.bin --video examples/media/user.mp4 --write_video output/user_result.avi --write_json output/ --display 0 --net_resolution "1312x736" --scale_number 8 --scale_gap 0.125
