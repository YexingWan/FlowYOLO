#!/bin/sh

sudo python3 main.py \
--task inference \
--yolo_config_path './config/head_yolov3.cfg' \
--yolo_resume './work/checkpoints_head/4000_weights/yolo_f.pth' \
--flow_model "FlowNet2CS" \
--flow_resume './work/checkpoints_head/4000_weights/flow.pth' \
--data_config_path "config/head_data.data" \
--conf_thres 0.96 \
--nms_thres 0.1
