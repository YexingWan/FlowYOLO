#!/bin/sh

yolo_config_path='./config/head_yolov3.cfg'
yolo_resume='./work/checkpoints_head/4000_weights/yolo_f.pth'
flow_model='FlowNet2CS'
flow_resume='./work/checkpoints_head/4000_weights/flow.pth'
train_batch_size='3'
data_config_path='./config/head_data.data'
conf_thre='0.96'
nms_thres='0.1'


sudo python3 main.py \
--task 'inference' \
--yolo_config_path "$yolo_config_path" \
--yolo_resume "$yolo_resume" \
--flow_model "$flow_model" \
--flow_resume "$flow_resume" \
--data_config_path "$data_config_path"
--conf_thres "$conf_thre" \
--nms_thres "$nms_thres"
