import os, sys, glob
import numpy as np
import cv2
from math import floor
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.transform import resize
import xml.etree.ElementTree as ET

from collections import defaultdict

# for predict
class VideoFile(Dataset):
    def __init__(self,args, image_size = 448, src = '',camera = False, gap = 1, start = 0, duration = -1):
        self.args = args
        self.camera = camera
        self.src = src if not self.camera else ""
        self.gap = gap if not self.camera else -1


        if (not self.camera) and src:
            ext = os.path.splitext(src)[-1]
            ext_set = ('.mkv', '.avi', '.mp4', '.rmvb', '.AVI', '.MKV', '.MP4')
            if os.path.isfile(src) and ext in ext_set:
                self.cap = cv2.VideoCapture(src)
        elif self.camera:
            self.cap = cv2.VideoCapture(0)
        else:
            print(sys.stderr, "ERROR: Video {} is not exist or with wrong path.".format(src))
            quit(1)

        # Video
        if not self.camera and self.cap.isOpened():
            self.frames_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

            # set range of video to infer
            start = 0 if start < 0 else start
            self.start = start if start < floor(self.frames_num/self.fps) else \
                {print(sys.stderr, "ERROR: Video only has {} seconds".format(self.frames_num/self.fps))&quit(1)}
            if duration<=0:
                self.duration = floor(self.frames_num/self.fps) - start
            elif start+duration > floor(self.frames_num/self.fps):
                print(sys.stderr, "ERROR: Video only has {} seconds".format(self.frames_num / self.fps)) & quit(1)
            else:
                self.duration = duration
            self.start_frame = floor(self.start * self.fps)

        # Camera
        elif self.camera and self.cap.isOpened():
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            # set range of video to infer
            self.start = 0
            if duration<=0:
                self.duration = -1
                self.frames_num = -1
            else:
                self.duration = duration
                self.frames_num = self.duration*self.fps
        else:
            print(sys.stderr, "ERROR: Video/Camera can not opened by VideoCapture.")
            quit(1)
            return

        self.img_shape = (image_size,image_size)
        args.inference_size = self.img_shape
        args.fps = self.fps
        args.frame_size = self.frame_size
        return


    def __getitem__(self, index):
        if not self.camera:
            index = index + self.start_frame
            # get 2 frame with gap set
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ret, img = self.cap.read()
        else:
            ret, img = self.cap.read()

        if ret:
            # convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print("conver to RGB")
            # print("dataset inner check 1 max:{}".format(img.max()))
            h, w, _ = img.shape
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding(127.5)
            input_img = np.pad(img, pad, 'constant', constant_values=127.5)
            # print("dataset inner check 2 max:{}".format(input_img.max()))
            # Resize
            # input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
            input_img = cv2.resize(input_img,self.img_shape)
            # print("dataset inner check 3 max:{}".format(input_img.max()))
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()


            return "ignore",input_img
        else:
            print(sys.stderr, "ERROR: try to get frames{} and {}. But video {} only has {} frames.".format(index,index+self.gap,self.src,self.frames_num))
            exit(1)




    def __len__(self):
        if not self.camera:
            return floor(self.duration * self.fps) - self.gap
        else:
            if self.frames_num>0:
                return self.frames_num
            else:
                return 9223372036854775807


# one sequence one dataset, for train
class SequenceImage(Dataset):
    def __init__(self, img_folder_path, img_size=448):
        self.files = sorted(glob.glob('%s/*.*' % img_folder_path))
        self.annotation = [p.replace("Data","Annotations").replace("JPEG","xml") for p in self.files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.classes_map = {
            '__background__' : 0,  # always index 0
            'n02691156':1,
            'n02419796':2,
            'n02131653':3,
            'n02834778':4,
            'n01503061':5,
            'n02924116':6,
            'n02958343':7,
            'n02402425':8,
            'n02084071':9,
            'n02121808':10,
            'n02503517':11,
            'n02118333':12,
            'n02510455':13,
            'n02342885':14,
            'n02374451':15,
            'n02129165':16,
            'n01674464':17,
            'n02484322':18,
            'n03790512':19,
            'n02324045':20,
            'n02509815':21,
            'n02411705':22,
            'n01726692':23,
            'n02355227':24,
            'n02129604':25,
            'n04468005':26,
            'n01662784':27,
            'n04530566':28,
            'n02062744':29,
            'n02391049':30
        }

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        #print("Image.open max:{}".format(img.max()))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding (127.5)
        input_img = np.pad(img, pad, 'constant', constant_values=127.5)
        padded_h, padded_w, _ = input_img.shape
        # Resize
        input_img = cv2.resize(input_img, self.img_shape)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.annotation[index % len(self.annotation)].rstrip()
        boxes = self.ProcessXMLAnnotation(label_path)
        filled_labels = np.zeros((self.max_objects, 5))
        if boxes is not None:
            for idx,box in enumerate(boxes):
                x1 = box["xmin"]
                x2 = box['xmax']
                y1 = box["ymin"]
                y2 = box["ymax"]
                x1 += pad[1][0]
                y1 += pad[0][0]
                x2 += pad[1][0]
                y2 += pad[0][0]
                center_x= float(((x1 + x2) / 2)) / float(padded_w)
                center_y = float(((y1 + y2) / 2)) / float(padded_h)
                scale_w = float(abs(x2 - x1)) / float(padded_w)
                scale_h = float(abs(y2 - y1)) / float(padded_h)
                filled_labels[idx] = np.array([center_x,center_y,scale_w,scale_h,self.classes_map[box["name"]]])
        filled_labels = torch.from_numpy(filled_labels)
        return input_img, filled_labels


    def __len__(self):
        return len(self.files)


    def ProcessXMLAnnotation(self, xml_file):
        """Process a single XML file containing a bounding box."""
        # pylint: disable=broad-except
        try:
            tree = ET.parse(xml_file)
        except Exception:
            print('Failed to parse: ' + xml_file, file=sys.stderr)
            return None
        # pylint: enable=broad-except
        root = tree.getroot()
        boxes = []
        for object in root.iter("object"):
            box = dict()
            # Grab the 'index' annotation.
            box_tree = object.find("bndbox")
            box["xmin"] = int(box_tree.find('xmin').text)
            box["ymin"] = int(box_tree.find('ymin').text)
            box["xmax"] = int(box_tree.find('xmax').text)
            box["ymax"] = int(box_tree.find('ymax').text)
            box["name"] = object.find("name").text
            boxes.append(box)
        return boxes



class dictDataset_coco_intersect_VID(Dataset):
    def __init__(self, dataset_list):
        print("sequence_num:{}".format(dataset_list))
        dataloader_list = [DataLoader(dataset_list[i], 1) for i in range(len(dataset_list))]
        max = 0
        self.iter_dict = dict()
        self.loader_dict = dict()

        # save each iter of loader in a dict with class index as key
        for idx,loader in enumerate(dataloader_list):
            # print("each_loader_len:{}".format(len(loader)))
            self.iter_dict[idx+1] = iter(loader)
            self.loader_dict[idx+1] = loader
            max  = len(loader) if len(loader) > max else max
        self.max_index = len(dataloader_list) * max
        self.num_loader = len(dataloader_list)


    def __getitem__(self, index):
        index = index % self.num_loader
        cur_iter = self.iter_dict[index+1]
        try:
            image, target = cur_iter.next()
            return index+1, torch.squeeze(image),torch.squeeze(target)
        except StopIteration:
            self.iter_dict[index] = iter(self.loader_dict[index])
            cur_iter = self.iter_dict[index]
            image, target = cur_iter.next()
            # if new iter, add 999 as a magic number
            return index+1+99999, torch.squeeze(image), torch.squeeze(target)

    def __len__(self):
        return self.max_index




# built list of datasets
def built_training_datasets(path):
    """
    1   airplane
    2   bear
    3   bicycle
    4   bird
    5   bus
    6   car
    7   dog
    8   cat
    9   elephant
    10  horse
    11  sheep
    12  train
    13  zebra
    """

    intersect = [1,3,4,5,6,7,9,10,11,15,22,26,30]
    folder_path = set()
    #print("traing_root_path:{}".format(path))

    p_list = ["ImageSets/VID/train_{}.txt".format(d) for d in intersect]


    for idx, f in enumerate([os.path.join(path,p_f) for p_f in p_list]):
        if os.path.isfile(f):
            with open(f,"r") as file:
                for line in file:
                    data_path = line.split(" ")[0]
                    #print("data_path:{}".format(data_path))
                    #class_folder_dict[idx+1].append(os.path.join(os.path.join(path,"Data/VID/train"),data_path))
                    folder_path.add(os.path.join(os.path.join(path,"Data/VID/train"),data_path))

    dataset_list = []
    # for p in glob.glob(os.path.join(path,"Data/VID/train/*")):
    #     for s_path in glob.glob(os.path.join(p, "*")):
    #print(folder_path)
    for p in folder_path:
        dataset_list.append(SequenceImage(p))

    return dataset_list
