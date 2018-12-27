import os, sys, glob, random
import numpy as np
import cv2
from math import floor
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET


VID_map = {
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


# for predict (video input)
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
            h, w, _ = img.shape
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding(127.5)
            input_img = np.pad(img, pad, 'constant', constant_values=127.5)

            input_img = cv2.resize(input_img,self.img_shape)

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


# for general train predict and test (sequence input) of one sequence
class SequenceImage(Dataset):
    def __init__(self, img_folder_path, img_size=448,map = VID_map):
        self.files = sorted(glob.glob('%s/*.*' % img_folder_path))
        self.annotation = [p.replace("Data","Annotations").replace("JPEG","xml") for p in self.files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.classes_map = map



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
        boxes = ProcessXMLAnnotation(label_path)
        filled_labels = np.zeros((self.max_objects, 5))


        if boxes is not None:
            for idx,box in enumerate(boxes):
                if self.classes_map[box["name"]] in self.ids:
                    x1 = box["xmin"]
                    x2 = box['xmax']
                    y1 = box["ymin"]
                    y2 = box["ymax"]
                    x1 += pad[1][0]
                    y1 += pad[0][0]
                    x2 += pad[1][0]
                    y2 += pad[0][0]

                    # x y w h is 0-1 scaled
                    center_x= float(((x1 + x2) / 2)) / float(padded_w)
                    center_y = float(((y1 + y2) / 2)) / float(padded_h)
                    scale_w = float(abs(x2 - x1)) / float(padded_w)
                    scale_h = float(abs(y2 - y1)) / float(padded_h)

                    # label is index in tensor, not the key in dict, index = key-1
                    filled_labels[idx] = np.array([center_x,center_y,scale_w,scale_h,self.classes_map[box["name"]]-1])

        filled_labels = torch.from_numpy(filled_labels)
        return input_img, filled_labels


    def __len__(self):
        return len(self.files)


# special designed Dataset for general train
class dictDataset(Dataset):
    def __init__(self, dataset_list):
        print("sequence_num:{}".format(len(dataset_list)))
        self.dataloader_list = [DataLoader(dataset_list[i], 1) for i in range(len(dataset_list))]
        max = 0
        self.iter_dict = dict()

        # save each iter of loader in a dict with index as key
        for idx,loader in enumerate(self.dataloader_list):
            # print("each_loader_len:{}".format(len(loader)))
            self.iter_dict[idx] = iter(loader)
            max  = len(loader) if len(loader) > max else max
        self.max_index = len(self.dataloader_list) * max
        self.num_loader = len(self.dataloader_list)


    def __getitem__(self, index):
        index = index % self.num_loader
        cur_iter = self.iter_dict[index]
        try:
            image, target = cur_iter.next()
            return index, torch.squeeze(image),torch.squeeze(target)
        except StopIteration:
            self.iter_dict[index] = iter(self.dataloader_list[index])
            cur_iter = self.iter_dict[index]
            image, target = cur_iter.next()
            # if new iter, add 999 as a magic number
            return index+99999, torch.squeeze(image), torch.squeeze(target)

    def __len__(self):
        return self.max_index



class SequenceImage_coco_intersect_VID(Dataset):
    def __init__(self, img_folder_path,class_ids, img_size=448):
        self.files = sorted(glob.glob('%s/*.*' % img_folder_path))
        self.annotation = [p.replace("Data","Annotations").replace("JPEG","xml") for p in self.files]
        self.img_shape = (img_size, img_size)
        self.ids = set(class_ids)
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

        self.reverse_classes_map = {
            0:'__background__',  # always index 0
            1:'n02691156',
            2:'n02419796',
            3:'n02131653',
            4:'n02834778',
            5:'n01503061',
            6:'n02924116',
            7:'n02958343',
            8:'n02402425',
            9:'n02084071',
            10:'n02121808',
            11:'n02503517',
            12:'n02118333',
            13:'n02510455',
            14:'n02342885',
            15:'n02374451',
            16:'n02129165',
            17:'n01674464',
            18:'n02484322',
            19:'n03790512',
            20:'n02324045',
            21:'n02509815',
            22:'n02411705',
            23:'n01726692',
            24:'n02355227',
            25:'n02129604',
            26:'n04468005',
            27:'n01662784',
            28:'n04530566',
            29:'n02062744',
            30:'n02391049'
        }

        self.origin_map_new_ids = dict(zip(class_ids,[i+1 for i in range(len(class_ids))]))


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
        boxes = ProcessXMLAnnotation(label_path)
        filled_labels = np.zeros((self.max_objects, 5))


        if boxes is not None:
            for idx,box in enumerate(boxes):
                if self.classes_map[box["name"]] in self.ids:
                    x1 = box["xmin"]
                    x2 = box['xmax']
                    y1 = box["ymin"]
                    y2 = box["ymax"]
                    x1 += pad[1][0]
                    y1 += pad[0][0]
                    x2 += pad[1][0]
                    y2 += pad[0][0]

                    # x y w h is 0-1 scaled
                    center_x= float(((x1 + x2) / 2)) / float(padded_w)
                    center_y = float(((y1 + y2) / 2)) / float(padded_h)
                    scale_w = float(abs(x2 - x1)) / float(padded_w)
                    scale_h = float(abs(y2 - y1)) / float(padded_h)

                    # label is index in tensor, not the key in dict, index = key-1
                    filled_labels[idx] = np.array([center_x,center_y,scale_w,scale_h,self.origin_map_new_ids[self.classes_map[box["name"]]]-1])

        filled_labels = torch.from_numpy(filled_labels)
        return input_img, filled_labels


    def __len__(self):
        return len(self.files)



def ProcessXMLAnnotation(xml_file):
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



def built_coco_intersect_VID_datasets(path,img_size):
    """
    origin_idx  idx class

    1   1   airplane
    3   2   bear
    4   3   bicycle
    5   4   bird
    6   5   bus
    7   6   car
    9   7   dog
    10  8   cat
    11  9   elephant
    15  10  horse
    22  11  sheep
    26  12  train
    30  13  zebra

    """

    intersect = [1,3,4,5,6,7,9,10,11,15,22,26,30]
    folder_path = set()

    p_list = ["ImageSets/VID/train_{}.txt".format(d) for d in intersect]

    for idx, f in enumerate([os.path.join(path,p_f) for p_f in p_list]):
        if os.path.isfile(f):
            with open(f,"r") as file:
                for line in file:
                    data_path = line.split(" ")[0]
                    folder_path.add(os.path.join(os.path.join(path,"Data/VID/train"),data_path))

    dataset_list = []
    #print(folder_path)
    for p in folder_path:
        dataset_list.append(SequenceImage_coco_intersect_VID(p,intersect,img_size=img_size))

    return dataset_list


# built list of num_sequence datasets
def built_VID_datasets(path = "/disk2/wanyx/ILSVRC2015",num_sequence:int = -1):
    print("path:{}".format(path))
    _tep = glob.glob(os.path.join(path,"Data/VID/train/")+"*")
    print("tep:{}".format(_tep))
    folder_path = []
    for p in _tep:
        folder_path.extend(glob.glob(os.path.join(p,"*")))
    print("sequence found:{}".format(len(folder_path)))
    num_sequence = len(folder_path) if num_sequence == -1 else num_sequence
    print("sample {} sequence".format(num_sequence))
    folder_path = random.sample(folder_path,num_sequence)


    dataset_list = []
    for p in folder_path:
        dataset_list.append(SequenceImage(p))
    return dataset_list



# built built list of dataloader (for val)
def built_dataloaders(datasets:list):
    loaders = []
    for dataset in datasets:
        loaders.append(DataLoader(dataset))
    return loaders
