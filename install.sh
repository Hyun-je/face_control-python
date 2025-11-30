#!/bin/bash

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

bash install_ola.sh
bash install_dmx.sh
bash install_face.sh