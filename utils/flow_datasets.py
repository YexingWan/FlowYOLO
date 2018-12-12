import torch
import torch.utils.data as data
import sys
import os, math, random
from os.path import *
import numpy as np
import utils.tools as tools
from glob import glob
import utils.frame_utils as frame_utils
import cv2
from math import floor

from scipy.misc import imread, imresize


#根据flow读数据，这里是有问题的，不用来来infer自己的数据，代码还要调整

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])


            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')


            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))
        return images, flow

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)


class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return images, flow

  def __len__(self):
    return self.size * self.replicates


class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.flo')) )
        for i in range(len(flows)):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return images, flow

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return images, flow

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))

    #[channel_idx，2，row_idx，col_idx]
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return images, torch.zeros((2,)+images.size()[-2:])

  def __len__(self):
    return self.size * self.replicates

class VideosInference:
    """
    src: root path of video files
    save: path for saving result (flows, flow_vis, flow_vis_video)
    """
    def __init__(self, src:str, save:str = "./infer/"):
        files_list = glob(os.path.join(src,"*"))
        self.files = [f for f in files_list if os.path.splitext(f)[-1] in ('.mkv','.avi','.mp4','.rmvb','.AVI','.MKV','.MP4')]
        self.files.sort()
        if not os.path.isdir(save):
            os.mkdir(save, mode=0o755)
        self.save = save
        self.idx = 0

    def __iter__(self):
        return self.files


    def __getitem__(self, index):
        return self.files[index]

    def __next__(self):
        if self.idx == len(self.files):
            raise StopIteration
        else:
            data = self.files[self.idx]
            self.idx += 1
            return data

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return self.files




class VideoFile(data.Dataset):
    def __init__(self,args,is_cropped = False, root = '',camera = False, gap = 1, start = 0, duration = -1):
        self.camera = camera
        if (not camera) and root:
            ext = os.path.splitext(root)[-1]
            ext_set = ('.mkv', '.avi', '.mp4', '.rmvb', '.AVI', '.MKV', '.MP4')
            if os.path.isfile(root) and ext in ext_set:
                self.cap = cv2.VideoCapture(root)
        elif camera:
            self.camera = True
            self.cap = cv2.VideoCapture(0)
        else:
            print(sys.stderr, "ERROR: Video {} is not exist or with wrong path.".format(root))
            quit(1)

        if not camera and self.cap.isOpened():
            self.frames_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_size = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        elif camera and self.cap.isOpened():
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_size = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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

        self.args = args
        self.src = root if not self.camera else ""
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.gap = gap if not self.camera else -1
        # render_size: size of input of model
        self.render_size = args.inference_size
        if (self.render_size[0] < 0):
            self.render_size[0] = self.frame_size[0]
        if (self.render_size[1] < 0):
            self.render_size[1] = self.frame_size[1]

        self.render_size[0] = int(((self.render_size[0]) // 64) * 64)
        self.render_size[1] = int(((self.render_size[1]) // 64) * 64)

        # # set the render size of img provide to net
        # if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
        #     self.frame_size[1] % 64):
        #     self.render_size[0] = int(((self.frame_size[0]) // 64) * 64)
        #     self.render_size[1] = int(((self.frame_size[1]) // 64) * 64)
        args.inference_size = tuple(self.render_size)
        return


    def __getitem__(self, index):
        if not self.camera:
            index = index + self.start_frame
            # get 2 frame with gap set
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ret1, img1 = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index+self.gap)
            ret2, img2 = self.cap.read()
        else:
            ret1, img1 = self.cap.read()
            ret2, img2 = self.cap.read()

        if ret1 and ret2:
            image_size = img1.shape[:2]
            rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            images = [rgb1, rgb2]
            # if require crop, the frame is random crop to crop_size (256*256 default)
            # in reference, is_crop default false
            #print("render_size:{}".format(self.render_size))
            if self.is_cropped:
                cropper = StaticRandomCrop(image_size, self.crop_size)
                images = list(map(cropper, images))
            else:
                # center crop to render_size
                # cropper = StaticCenterCrop(image_size, self.render_size)
                # replace center crop to resize directly
                images = [cv2.resize(img,dsize=(self.render_size[1],self.render_size[0]),interpolation=cv2.INTER_LINEAR) for img in images]

            #print("final_input_size:{}".format(images[0].shape))

            # images[channel,2,row_idx,col_idx]
            images = np.array(images).transpose(3, 0, 1, 2)
            images = torch.from_numpy(images.astype(np.float32))
            fake_flow = torch.zeros((2,)+images.size()[-2:])
            #print("fack_flow_size:{}".format(fake_flow.size()))
            return images, fake_flow
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







'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
