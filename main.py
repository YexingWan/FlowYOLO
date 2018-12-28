import matplotlib as mpl
mpl.use('Agg')
import models
import torch
import cv2
from torch.utils.data import DataLoader
import tqdm

import argparse, os
import numpy as np
import random

from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PIL import Image

from utils import utils, datasets
from utils.parse_config import parse_data_config



def built_args():
    parser = argparse.ArgumentParser(argparse.ArgumentParser)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task', choices=['train','test','inference'], default='inference')
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument("--save_video", action='store_true', default=False)
    parser.add_argument("--camera", action='store_true',default=False,help='inference from camera')
    parser.add_argument("--data_config_path", type=str, default="config/data.data", help="path to data config file")


    parser.add_argument('--validation_frequency', type=int, default=10000, help='validate every n batches')
    parser.add_argument('--validation_n_sequence', type=int, default=5)
    parser.add_argument('--validation_batch_size', type=int, default=1, help="Do not support mini-batch validation currently")



    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--train_batch_size', type=int, default=8)

    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001,help="Learning rate for edn-to-end traning")
    parser.add_argument('--momentum', type=float, default=0.9,help="Momentum for edn-to-end traning")
    parser.add_argument('--decay', type=float, default=0.005, help="Weight decay for edn-to-end traning")
    parser.add_argument('--saving_checkpoint_interval', type=int, default=10000, help="Number of batches for saving weight")


    parser.add_argument('--inference_batch_size', type=int, default=1, help="Do not support mini-batch inference currently")
    parser.add_argument('--inference_n_batches', type=int, default=-1,
                        help='Number of min-batches for inference. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--inference_size', type=int, default=448,
                        help='image size for inference, image will resize to such size, should be divided by 64')
    #parser.add_argument('--aggregate_range', type=int, default=2)


    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving all result')
    # parser.add_argument('--skip_training', action='store_true')
    # parser.add_argument('--skip_validation', action='store_true')

    #--------------------------------------flownet--------------------------------------------
    parser.add_argument('--flow_model', default='FlowNet2CS')
    parser.add_argument('--flow_resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    #--------------------------------YOLO-----------------------------------------------------

    parser.add_argument("--yolo_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
    parser.add_argument("--yolo_resume", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    args = parser.parse_args()

    # model class
    args.flow_model_class = utils.module_to_dict(models)[args.flow_model]
    args.yolo_model_class = utils.module_to_dict(models)["Darknet"]

    # data config
    data_config = parse_data_config(args.data_config_path)
    args.data_test_path = data_config["valid"]
    args.data_train_path = data_config["train"]
    args.data_infer_path = data_config["infer"]
    args.data_num_classes = int(data_config["classes"])
    args.data_names_path = data_config["names"]
    args.rgb_max=int(data_config["rgb_max"])

    # history things, ignore
    args.fp16=None

    return args


def train(args):

    flow_yolo = models.FlowYOLO(args)

    flow_yolo.train()

    # TODO：
    # deprecated
    # use new weigth
    flow_yolo.load_weights(args.flow_resume, args.yolo_resume)

    for p in flow_yolo.parameters():
        p.requires_grad = True

    # freeze flownet
    for p in flow_yolo.flow_model.parameters():
        p.requires_grad = False

    # set model cuda
    if torch.cuda.is_available() and args.use_cuda:
        number_gpus = torch.cuda.device_count()
        if number_gpus > 0:
            print("GPU_NUMBER:{}".format(number_gpus))
            # use muti-GPU
            args.train_batch_size *= number_gpus
            #flow_yolo = nn.parallel.DataParallel(flow_yolo, device_ids=list(range(number_gpus)))
            flow_yolo.set_multi_gpus(list(range(number_gpus)))


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, flow_yolo.parameters()),lr=1e-3)
    # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, flow_yolo.parameters()))

    ###########bulit dataset for traning##########

    train_dataset_list, val_dataset_list = datasets.built_VID_datasets(args.data_train_path,1/4)
    #dataset_list = datasets.built_coco_intersect_VID_datasets(path = args.data_train_path)

    train_final_dataset = datasets.dictDataset(train_dataset_list)

    train_final_loader = DataLoader(train_final_dataset,args.train_batch_size,shuffle=True)

    val_final_loader_list = datasets.built_dataloaders(val_dataset_list,1,False)

    ########2 dictionaries keep for flow##########

    feature_dict = dict(zip([i for i in range(len(train_dataset_list))],[None for _ in range(len(train_dataset_list))]))

    last_frame_dict = dict(zip([i for i in range(len(train_dataset_list))],[None for _ in range(len(train_dataset_list))]))

    cur_batch = 0
    total_batch = len(train_final_loader)
    print("total_batch:{}".format(total_batch))
    print("total_epoch:{}".format(args.total_epochs))

    # training loops
    for epoch in range(args.total_epochs):

        for b, (seq_index, images, targets) in enumerate(train_final_loader):
            """
            targets.shape:[batch,50,5(x,t,w,h,class)]
                x: 0-1 scaled, centered, padded
                y: 0-1 scaled, centered, padded
                w: 0-1 scaled, padded
                h: 0-1 scaled, padded
                class: int, index of class
            
            seq_index: sequence indexes
            
            images: input batched images, 255 base, inferenced size(448)
            """

            # initialize optimizer and list for storing
            optimizer.zero_grad()
            last_feature =  []
            flow_input = []

            for i,t_idx in enumerate(seq_index):
                idx = t_idx.item()
                # if current frame is the first frame (flag by +99999), initialize element two dict
                if idx >= 99999:
                    idx -= 99999
                    last_frame_dict[idx] = None
                    feature_dict[idx] = None


            for i, t_idx in enumerate(seq_index):
                idx = t_idx.item()
                if idx >= 99999:
                    idx -= 99999
                # if current frame is the first frame, set two list for flow-warp to None
                # for simplify, once one of frame is the first frame in batch, all frame train as first frame (without flow)
                if last_frame_dict[idx] is None or feature_dict[idx] is None:
                    flow_input = None
                    last_feature = None
                    break
                # else, built input of flownet and prepare the feature for warping
                else:
                    # notice: flow input is current frame to last_frame
                    flow_input.append(torch.stack([images[i],last_frame_dict[idx]]).permute(1, 0, 2, 3))
                    last_feature.append(feature_dict[idx])

            # built final flow_input
            if flow_input is not None:
                flow_input = torch.stack(flow_input)

            # update the last_frame_dict
            for i, t_idx in enumerate(seq_index):
                idx = t_idx.item()
                if idx >= 99999:
                    idx -= 99999
                last_frame_dict[idx] = images[i]

            # set input data cuda
            if args.use_cuda:
                flow_input = flow_input.cuda() if flow_input is not None else None
                last_feature = [[f.cuda() for f in l] for l in last_feature] if last_feature is not None else None
                images = images.cuda()

            # forward operation returns losses(dictionary) and list of list of feature for next frame warping.
            # s = time.time()
            # feature is list of list of cup.tensor
            losses, feature = flow_yolo(flow_input,images,last_feature,targets)
            # e = time.time()
            # print("train forward time:{}".format(e-s))


            for i, t_idx in enumerate(seq_index):
                idx = t_idx.item()
                if idx >= 99999:
                    idx -= 99999
                _fe = []
                # unite the output feature from model to save memory
                for f in feature[i]:
                    _fe.append(f.detach().cpu())
                # save features in dictionary by sequence index
                feature_dict[idx] = _fe

            # get "loss" tensor and backward to get grad
            losses["loss"].backward()

            # update weights
            optimizer.step()

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    args.total_epochs,
                    cur_batch,
                    total_batch,
                    losses["x"],
                    losses["y"],
                    losses["w"],
                    losses["h"],
                    losses["conf"],
                    losses["cls"],
                    losses["loss"].item(),
                    losses["recall"],
                    losses["precision"],
                )
            )
            cur_batch += 1
            # clear unused memory
            torch.cuda.empty_cache()
            # save checkpoint for 10000 batches


            #save checkpoint and validation
            if cur_batch % args.validation_frequency == 0:
                print("validation in {} batch".format(cur_batch))
                test(flow_yolo,val_final_loader_list,args)
                flow_yolo.train()

            if cur_batch % args.saving_checkpoint_interval == 0:
                final_save_path = "%s/%d_weights" % (os.path.join(args.save,"checkpoints"), cur_batch)
                flow_yolo.save_weights(final_save_path)
                print("Save success. Weight save in {}.".format(final_save_path))
    print("Done!")


def test(model,dataloader_list:list,args):

    # initialize
    model.eval()
    args.validation_batch_size = 1
    num_classes = args.data_num_classes
    args.inference_batch_size = 1

    print("Compute mAP...")

    all_detections = []
    all_annotations = []


    for loader_idx, dataloader in enumerate(tqdm.tqdm(dataloader_list,desc="Sequence list")):
        last_frame = None
        last_feature = None

        for batch_i, (images, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            """
            targets.shape:[batch,50,5(x,y,w,h,class)]
                x: 0-1 scaled, centered, padded
                y: 0-1 scaled, centered, padded
                w: 0-1 scaled, padded
                h: 0-1 scaled, padded
                class: int, index of class

            seq_index: sequence indexes

            images: input batched images, 255 base [batch_size, c, h, w], inferenced size(448)           
            """

            #assert(images.shape[0] == 1)
            flow_input = torch.unsqueeze(torch.stack([images[0], last_frame]).permute(1, 0, 2, 3),dim=0) if last_frame is not None else None
            last_frame = images[0]

            # set cuda
            if args.use_cuda:
                flow_input = flow_input.cuda() if flow_input is not None else None
                images = images.cuda()


            with torch.no_grad():
                # 这里的output是经过预处理的output（在yoloLayer中预处理），[nB,number of anchory, bbox_attrs(5+num_classes)]
                # 其中bbox_attrs为 x y w h conf cls_conf的predict
                # 其中xywh为每个box对应resized image的 "unscaled" x，y坐标（center）和w，h大小
                # 注意这里是 w，h 不是 h，w
                outputs, features = model(flow_input = flow_input, data = images, last_feature = last_feature)
                last_feature = features

                # 经过nms以后的outputs是一个list of Tensor，每个Tensor代表一张图的prediction
                # tensor的shape为[num_selected_boxed, 7(x1, y1, x2, y2, obj_conf, class_conf, class_pred)] unscaled
                outputs = utils.non_max_suppression(outputs, 80, conf_thres=args.conf_thres, nms_thres=args.nms_thres)

            for output, annotations in zip(outputs, targets):
                # annotations也是一个Tensor[50,5(x,y,w,h,class)] scaled


                # 结果保存，每张图一个list，list里每个class对应一个nparray，array的shape为[num_box,5(x1, y1, x2, y2, obj_conf)]
                # all_detections 是 list of list (each image) of array(each class)
                all_detections.append([np.array([]) for _ in range(num_classes)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(num_classes):
                        #每个predict box的shape为[5(x1, y1, x2, y2, obj_conf)]
                        all_detections[-1][label] = pred_boxes[pred_labels == label]

                all_annotations.append([np.array([]) for _ in range(num_classes)])
                if any(annotations[:, -1] > 0):
                    annotation_labels = annotations[annotations[:, -1] > 0, -1].numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, :4].numpy()

                    # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= args.inference_size

                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions = {}
    # for each class
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        # for each image
        for i in range(len(all_annotations)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []


            # for each detected box
            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = utils.bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= args.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = utils.compute_ap(recall, precision)
        average_precisions[label] = average_precision

    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print("+ Class '{}' - AP: {}".format(c,ap))

    mAP = np.mean(list(average_precisions.values()))
    print("mAP: {}".format(mAP))


def inference(args):
    # built module
    flow_yolo = models.FlowYOLO(args)

    # TODO：
    # deprecated
    # use new weigth
    flow_yolo.load_weights(args.flow_resume, args.yolo_resume)

    # set cuda
    if torch.cuda.is_available() and args.use_cuda:
        number_gpus=torch.cuda.device_count()
        if number_gpus > 0:
            print("GPU_NUMBER:{}".format(number_gpus))
            flow_yolo.set_multi_gpus(gpu_id_list=list(range(number_gpus)))

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
        args.inference_batch_size = 1
    else:
        dataset,_ = datasets.SequenceImage(args.data_infer_path,None)
        args.inference_batch_size = 1
        cap = None
        v_writer= None

    # init data loader
    dataloader = DataLoader(dataset,batch_size=args.inference_batch_size,shuffle=False)

    last_feature = None
    last_frame = None

    # for each batch, input_imgs is 0-255 [b,c,h,w]
    for batch_i, input_imgs in enumerate(dataloader):

        flow_input = torch.unsqueeze(torch.stack([input_imgs[0], last_frame]).permute(1, 0, 2, 3),0) if last_frame is not None else None
        last_frame = input_imgs[0]

        if args.use_cuda:
            flow_input = flow_input.cuda() if flow_input is not None else None
            input_imgs = input_imgs.cuda()

        # Get detections
        with torch.no_grad():
            detections, features = flow_yolo(flow_input = flow_input,
                                             data = input_imgs,
                                             last_feature = last_feature)
            detections = utils.non_max_suppression(detections, len(classes), args.conf_thres, args.nms_thres)
            # features is a list of list
            last_feature = features

        # Save image and detections depends on type of source
        if cap is not None and v_writer is not None:
            v_writer = draw_and_save(args,
                          [cap.read()[1] for _ in range(args.inference_batch_size)],
                          detections,
                          classes,
                          v_writer=v_writer)
        else:
            draw_and_save(args,
                          np.transpose(last_frame.numpy(), (1, 2, 0)),
                          detections,
                          classes)
    v_writer.release()


def draw_and_save(args,source,img_detections,classes,v_writer = None):

    # get the colormap instance then get 20 colors
    cmap = plt.get_cmap('hot')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\nSaving result:')
    # Iterate through images and save plot of detections
    for img_i, (source, detections) in enumerate(zip(source, img_detections)):

        # Create plot by path
        if isinstance(source,str):
            img = np.array(Image.open(source))
        # or source is image itself
        else:
            img = source

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (args.inference_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (args.inference_size / max(img.shape))
        #print("pad_x:%d pad_y:%d " % (pad_x, pad_y))

        # Image height and width after padding is removed
        unpad_h = args.inference_size - pad_y
        unpad_w = args.inference_size - pad_x

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
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
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
    elif task == "inference":
        inference(args)


"""
python3 main.py --task inference --yolo_config_path "./config/yolov3.cfg" --yolo_resume "../yolo_weight/yolov3.pth" --flow_model "FlowNet2CS" --flow_resume "../flow_weight/FlowNet2-CS_checkpoint.pth"

python3 main.py --task train --yolo_config_path "./config/yolov3.cfg" --yolo_resume "../yolo_weight/yolov3.pth" --flow_model "FlowNet2CS" --flow_resume "../flow_weight/FlowNet2-CS_checkpoint.pth" --train_batch_size 2
"""

if __name__ == "__main__":
    args = built_args()
    if not os.path.isdir(args.save):
        os.mkdir(args.save)
    if not os.path.isdir(os.path.join(args.save,"checkpoints")):
        os.mkdir(os.path.join(args.save,"checkpoints"))
    assert(args.inference_size % 64 == 0)
    main(args,task = args.task)
