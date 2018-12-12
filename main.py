import yolo
import models
import flownet
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *
import time

import flownet.flowlib as flowlib
import flownet.models, flownet.losses, flownet.datasets
from flownet.utils import flow_utils, tools

import yolo.models
from yolo.utils.parse_config import parse_data_config, parse_model_config


def built_args():
    parser = argparse.ArgumentParser(argparse.ArgumentParser)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task', choices=['train','test','inference'], default='inference')
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument("--batched_n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--save_video", action='store_true', default=False)
    parser.add_argument("--camera", action='store_true',default=False,help='inference from camera')
    parser.add_argument("--data_config_path", type=str, default="config/train.data", help="path to data config file")

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument("--validation_checkpoint", type=int, default=5, help="interval between saving model weights, default saved at each validation ")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument('--crop_size', type=int, nargs='+', default=[416, 416],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001,help="Learning rate for edn-to-end traning")
    parser.add_argument('--momentum', type=float, default=0.9,help="Momentum for edn-to-end traning")
    parser.add_argument('--decay', type=float, default=0.005, help="Weight decay for edn-to-end traning")


    parser.add_argument('--inference_batch_size', type=int, default=8)
    parser.add_argument('--inference_n_batches', type=int, default=-1,
                        help='Number of min-batches for inference. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[416, 416],
                        help='image size for inference by flownet, image will resize to such size')
    parser.add_argument('--aggregate_range', type=int, default=2)


    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving all result')
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')


    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    #--------------------------------------flownet--------------------------------------------

    parser.add_argument('--flow_resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows from flownet to file', default=False)

    #--------------------------------YOLO-----------------------------------------------------

    parser.add_argument("--yolo_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--yolo_resume", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    args = parser.parse_args()

    # model class
    args.flow_model_class = tools.module_to_dict(models)[args.flow_model]
    args.yolo_model_class = tools.module_to_dict(models)["Darknet"]

    # data config
    data_config = parse_data_config(args.data_config_path)
    args.data_test_path = data_config["valid"]
    args.data_train_path = data_config["train"]
    args.data_num_classes = int(data_config["classes"])
    args.data_names_path = data_config["names"]
    args.data_infer_path = data_config["infer"]

    return args


def train(args):


    pass

def test(args):

    pass


def inference(args):
    dataloader = DataLoader()
    flow_yolo = models.FlowYOLO(args)
    flow_yolo.load_weights(args.flow_resume, args.yolo_resume)
    flow_yolo.eval()
    if torch.cuda.is_available() and args.use_cuda:
        number_gpus=torch.cuda.device_count()
        if number_gpus > 0:
            flow_yolo = nn.parallel.DataParallel(flow_yolo, device_ids=list(range(args.number_gpus)))
            flow_yolo = flow_yolo.cuda()
            torch.cuda.manual_seed(args.seed)






    pass

def main(args,task):
    if task == "train":
        train(args)
    elif task == "test":
        test(args)
    elif task == "inference":
        inference(args)



if __name__ == "__main__":
    args = built_args()
    main(args,task = args.task)
