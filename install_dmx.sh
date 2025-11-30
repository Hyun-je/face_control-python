#!/bin/bash

ola_patch -u 0 -d 8 -p 1

/home/pi/miniconda3/bin/conda create -y -n dmx python=3.8

/home/pi/miniconda3/envs/dmx/bin/python -m pip install ola
/home/pi/miniconda3/envs/dmx/bin/python -m pip install python-osc==1.8.3

sudo bash service/dmx/install.sh