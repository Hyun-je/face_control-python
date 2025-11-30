#!/bin/bash

conda create -y -n face python=3.9

/home/pi/miniconda3/envs/face/bin/python -m pip install mediapipe
/home/pi/miniconda3/envs/face/bin/python -m pip install opencv-python==4.6.0.66
/home/pi/miniconda3/envs/face/bin/python -m pip install numpy==1.26.4
/home/pi/miniconda3/envs/face/bin/python -m pip install python-osc==1.8.3
/home/pi/miniconda3/envs/face/bin/python -m pip install FaceAnalyzer==0.1.33
/home/pi/miniconda3/envs/face/bin/python -m pip install pyyaml

sudo bash service/face/install.sh