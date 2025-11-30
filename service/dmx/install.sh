#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
USER_SERVICE_DIR=/etc/systemd/system
SERVICE_NAME=dmx

mkdir $USER_SERVICE_DIR

echo "Create service file : $SERVICE_NAME.service"
printf "\
[Unit]
Requires=ola.service
After=ola.service

[Timer]
OnBootSec=5s

[Service]
Type=simple
ExecStart=/home/pi/miniconda3/envs/dmx/bin/python $SCRIPT_DIR/$SERVICE_NAME.py
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
