#!/bin/sh

yolo_config_path='./config/head_yolov3.cfg'
yolo_resume='./work/checkpoints/10000_weights/yolo_f.pth'
flow_model='FlowNet2CS'
flow_resume='./work/checkpoints/10000_weights/flow.pth'
train_batch_size=3
data_config_path='./config/head_data.data'

python3 ./main.py \
--task 'train' \
--yolo_config_path "$yolo_config_path" \
--yolo_resume "$yolo_resume" \
--flow_model 'FlowNet2CS' \
--flow_resume "$flow_resume" \
--train_batch_size "$train_batch_size" \
--data_config_path "$data_config_path"



