#!/bin/bash

mkdir output

./build/examples/openpose/openpose.bin --video examples/media/user_000.mp4 --write_json output/ --display 0 --render_pose 0 --net_resolution "1312x736" --scale_number 8 --scale_gap 0.125
