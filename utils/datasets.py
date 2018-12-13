import os, sys, glob
import numpy as np
import cv2
from math import floor
import torch

from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize



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
            h, w, _ = img.shape
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding(127.5)
            input_img = np.pad(img, pad, 'constant', constant_values=127.5)
            # Resize
            input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
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



# one sequence one dataset
class SequenceImage(Dataset):
    def __init__(self, folder_path, img_size=448):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding (127.5)
        input_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # Resize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return [img_path], input_img

    def __len__(self):
        return len(self.files)





# Traning dataset
class ImagenetVID(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
