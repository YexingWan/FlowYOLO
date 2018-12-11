#!/usr/bin/env bash


#parameter
model="FlowNet2CSS"
checkpoint="/home/user/wanyx/fn_weight/FlowNet2-CSS_checkpoint.pth.tar"
#inference_datase="MpiSintelClean/MpiSintelFinal/FlyingChairs/FlyingThingsClean/FlyingThingsFinal/VideoFile/ImagesFromFolder/ChairsSDHomTrain/ChairsSDHomTest"
inference_dataset="VideoFile"
inference_root="/home/user/wanyx/vedio/HCVR_ch2_main_20181005180000_20181005190000.avi"




# if infer video, number of worker should be 1 or strike my deadlock
python main.py \
--inference \
--save_video \
--model "$model" \
--number_workers 0 \
--inference_dataset "$inference_dataset" \
--inference_dataset_root "$inference_root" \
--inference_dataset_gap 1 \
--inference_batch_size 16 \
--inference_dataset_start 233 \
--inference_dataset_duration 10 \
--resume "$checkpoint"
