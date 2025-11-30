#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
USER_SERVICE_DIR=/etc/systemd/system
SERVICE_NAME=face

mkdir $USER_SERVICE_DIR

echo "Create service file : $SERVICE_NAME.service"
printf "\
[Unit]
Requires=dmx.service
After=dmx.service

[Timer]
OnBootSec=5s

[Service]
Type=simple
ExecStart=/home/pi/miniconda3/envs/face/bin/python $SCRIPT_DIR/$SERVICE_NAME.py --config /home/pi/face_control-python/service/face/config.yaml
User=pi
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
" > $USER_SERVICE_DIR/$SERVICE_NAME.service

chmod +x $USER_SERVICE_DIR/$SERVICE_NAME.service

echo "Start service : $SERVICE_NAME.service"
systemctl daemon-reload
systemctl reenable $SERVICE_NAME.service
systemctl restart $SERVICE_NAME.service
