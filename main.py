import matplotlib as mpl
mpl.use('Agg')
import models
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
import random
import time

from matplotlib.ticker import NullLocator
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PIL import Image

import flownet.flowlib as flowlib
import flownet.models, flownet.losses, flownet.datasets
from utils import flow_utils, tools, utils, datasets
from utils.parse_config import parse_data_config, parse_model_config



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
    parser.add_argument("--data_config_path", type=str, default="config/data.data", help="path to data config file")

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
    parser.add_argument('--crop_size', type=int, nargs='+', default=[448, 448],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--learning_rate', type=float, default=0.001,help="Learning rate for edn-to-end traning")
    parser.add_argument('--momentum', type=float, default=0.9,help="Momentum for edn-to-end traning")
    parser.add_argument('--decay', type=float, default=0.005, help="Weight decay for edn-to-end traning")


    parser.add_argument('--inference_batch_size', type=int, default=8)
    parser.add_argument('--inference_n_batches', type=int, default=-1,
                        help='Number of min-batches for inference. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[448, 448],
                        help='image size for inference by flownet, image will resize to such size')
    parser.add_argument('--aggregate_range', type=int, default=2)


    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving all result')
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')


    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    #--------------------------------------flownet--------------------------------------------
    parser.add_argument('--flow_model', default='FlowNet2CS')
    parser.add_argument('--flow_resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows from flownet to file', default=False)

    #--------------------------------YOLO-----------------------------------------------------

    parser.add_argument("--yolo_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--yolo_resume", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.2, help="iou threshold required to qualify as detected")
    args = parser.parse_args()

    # model class
    args.flow_model_class = tools.module_to_dict(models)[args.flow_model]
    args.yolo_model_class = tools.module_to_dict(models)["Darknet"]

    # data config
    data_config = parse_data_config(args.data_config_path)
    args.data_test_path = data_config["valid"]
    args.data_train_path = data_config["train"]
    args.data_infer_path = data_config["infer"]
    args.data_num_classes = int(data_config["classes"])
    args.data_names_path = data_config["names"]
    args.rgb_max=int(data_config["rgb_max"])

    args.fp16=None

    return args


def train(args):


    pass

def test(args):

    pass


def inference(args):
    # built module
    flow_yolo = models.FlowYOLO(args)

    #load weight
    flow_yolo.load_weights(args.flow_resume, args.yolo_resume)

    # set cuda
    if torch.cuda.is_available() and args.use_cuda:
        number_gpus=torch.cuda.device_count()
        if number_gpus > 0:
            print("GPU_NUMBER:{}".format(number_gpus))
            flow_yolo = nn.parallel.DataParallel(flow_yolo, device_ids=list(range(number_gpus)))
            flow_yolo.cuda()

    # set to eval mode
    flow_yolo.eval()

    # load classes labels
    classes = utils.load_classes(args.data_names_path)  # Extracts class labels from file

    # init dataset and load cap and writer if source is video
    if args.camera or (os.path.splitext(args.data_infer_path)[-1] in ('.mkv', '.avi', '.mp4', '.rmvb', '.AVI', '.MKV', '.MP4') and os.path.isfile(args.data_infer_path)):
        dataset = datasets.VideoFile(args,src = args.data_infer_path, camera =args.camera, start=0, duration=10)
        print("video dataset!")
        if args.camera:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.data_infer_path)

        v_writer = cv2.VideoWriter()

    else:
        dataset = datasets.SequenceImage(folder_path=args.data_infer_path)
        cap = None
        v_writer= None

    # init data loader
    dataloader = DataLoader(dataset,batch_size=args.inference_batch_size)

    # for each batch
    for batch_i, (paths, input_imgs) in enumerate(dataloader):

        #print("paths_type:{}".format(type(paths)))
        #print("input_imgs_type:{}".format(type(input_imgs)))


        if args.use_cuda:
            input_imgs = input_imgs.cuda()

        # Get detections
        print("outer check data max:{}".format(input_imgs.max()))
        s = time.time()
        print("input_tensor_shape:{}".format(input_imgs.shape))
        with torch.no_grad():
            detections = flow_yolo(input_imgs)
            print("thred:{}".format(args.conf_thres))
            detections = utils.non_max_suppression(detections, len(classes), args.conf_thres, args.nms_thres)

        e = time.time()
        print("forward_time:{}s".format(e-s))

        # Save image and detections depends on type of source
        if cap is not None and v_writer is not None:
            print("save result as video")
            v_writer = draw_and_save(args,
                          [cap.read()[1] for _ in range(args.inference_batch_size)],
                          detections,
                          classes,
                          v_writer=v_writer)
        else:
            if paths:
                print("save result pre pic")
                draw_and_save(args,
                              paths,
                              detections,
                              classes)
            else:
                print(sys.stderr,"Error: something wrong with dataloader, loss image path.")


def draw_and_save(args,source,img_detections,classes,v_writer = None):
    cmap = plt.get_cmap('hot')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    print('\nSaving result:')
    # Iterate through images and save plot of detections
    for img_i, (source, detections) in enumerate(zip(source, img_detections)):

        #print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        if isinstance(source,str):
            img = np.array(Image.open(source))
        else:
            img = source
        #print("h:%d w:%d " % (img.shape[0], img.shape[1]))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (args.inference_size[1] / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (args.inference_size[0] / max(img.shape))
        #print("pad_x:%d pad_y:%d " % (pad_x, pad_y))

        # Image height and width after padding is removed
        unpad_h = args.inference_size[0] - pad_y
        unpad_w = args.inference_size[1] - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                         edgecolor=color,
                                         facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        if v_writer is not None:
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if v_writer.isOpened():
            # Now we can save it to a numpy array.
                v_writer.write(data)
            else:
                v_writer = cv2.VideoWriter("output/result.avi",
                                           apiPreference=cv2.CAP_ANY,
                                           fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                           fps=int(args.fps),
                                           frameSize=(data.shape[1],data.shape[0]))
                print("data_shape:{}".format(data.shape))
                v_writer.write(data)

        else:
            plt.savefig('output/%06d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
            plt.close()
    return v_writer

def main(args,task):
    if task == "train":
        train(args)
    elif task == "test":
        test(args)
    elif task == "inference":
        inference(args)


"""
python3 main.py --task inference --yolo_config_path "./config/yolov3.cfg" --yolo_resume "../yolo_weight/yolov3.pth" --flow_model "FlowNet2CS" --flow_resume "../flow_weight/FlowNet2-CS_checkpoint.pth"
"""

if __name__ == "__main__":
    args = built_args()
    main(args,task = args.task)
