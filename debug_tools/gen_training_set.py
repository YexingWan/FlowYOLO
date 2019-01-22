import numpy as np
import sys,os
import cv2
from glob import glob
import xml.etree.ElementTree as ET


caffe_root = 'Caffe for model'
sys.path.insert(0, caffe_root + 'python')

import caffe

net_file= 'example/MobileNetSSD_deploy.prototxt'
caffe_model='example/MobileNetSSD_deploy.caffemodel'

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)
# CLASSES = ('background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('background', 'head', 'face')




'''
out: tuple of tuple of (xmax,xmin,ymax,ymin,class_label)
info: dictionary with items <folder:str, filename:str, size:(w,h)>
'''
def write_annotation(info,out,path):
    root = ET.Element('annotation')
    ET.SubElement(root,'folder').text = info['folder']
    ET.SubElement(root,'filename').text = info['filename']
    size = ET.SubElement(root,'size')
    object = ET.SubElement(root, 'object')

    ET.SubElement(size,'width').text = str(info['size'][0])
    ET.SubElement(size,'height').text = str(info['size'][1])


    for box in out:
        ET.SubElement(object,'name').text = box[4]
        budbox = ET.SubElement(object, 'budbox')
        ET.SubElement(budbox, 'xmax').text = str(box[0])
        ET.SubElement(budbox, 'xmin').text = str(box[1])
        ET.SubElement(budbox, 'ymax').text = str(box[2])
        ET.SubElement(budbox, 'ymin').text = str(box[3])
    tree = ET.ElementTree(root)
    tree.write(os.path.join(path,info['filename'])+'.xml')
    print("annotation parseing on: "+os.path.join(path,info['filename'])+'.xml')


def preprocess(src):
    img = cv2.resize(src, (416,416))
    img = img - 127.5
    img = img * 0.007843
    return img

# default as only one image pre batch, postprocess the output of model
def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]

    # absolute xmax,xmin, yman, ymin
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    # cls index
    cls = out['detection_out'][0,0,:,1]

    # confidence
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(img_file,img_name, anno_path = None, save_path = None):
    #origimg = cv2.imread(imgfile)
    #origimg = cv2.imread(imgfile)
    print("detecting: "+img_name)
    origimg = img_file
    img_dir, img_name = os.path.split(img_name)
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    #print(img.shape)


    # detect object from images, out is the  model
    net.blobs['data'].data[...] = img
    out = net.forward()




    box, conf, cls = postprocess(origimg, out)
    box_ann = []
    #info = dict([('folder',test_dir),('filename',imgfile),('size',(img.shape[1],img.shape[0]))])
    for i in range(len(box)):
        if box[i][0]== - origimg.shape[1] and box[i][1] == -origimg.shape[0]:
            print('no detected boxes')
            continue
        if save_path is not None:
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            p3 = (max(p1[0], 15), max(p1[1], 15))
            if int(cls[i]) == 1:
                color = (0,255,0)
                cv2.rectangle(origimg, p1, p2, color)
                title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
                cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, color, 1)
        if int(cls[i]==1):
            box_ann.append((box[i][2], box[i][0], box[i][3], box[i][1], CLASSES[int(cls[i])])) #xmax,xmin, yman, ymin, label
    if len(box_ann) == 0:
        print('no detected heads.')
    if anno_path is not None and os.path.isdir(anno_path) :
        info = dict([('folder',img_dir),('filename',os.path.splitext(img_name)[0]),('size',(origimg.shape[1],origimg.shape[0]))])
        #print('info:')
        #print(info)
        write_annotation(info, box_ann, anno_path)
    print('number of detected head(s): '+ str(len(box_ann)))

    if save_path is not None and os.path.isdir(save_path) :
        cv2.imwrite(os.path.join(save_path,img_name), origimg)

    return True


def gen(video_file):
    print("operating "+video_file)
    cap = cv2.VideoCapture(video_file)
    save_dir = os.path.splitext(video_file)[0]
    pic_dir = save_dir+'/images/'
    anno_dir = save_dir+'/Annotation/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    if not os.path.isdir(anno_dir):
        os.mkdir(anno_dir)
    idx = 0
    while cap.isOpened():
        if idx % 512 == 0:
            cur_pic_dir = pic_dir + str(idx//512) + '/'
            cur_anno_dir = anno_dir + str(idx//512)+ '/'
            cur_save_dir = save_dir +'/' + str(idx//512)+ '/'
        if not os.path.isdir(cur_pic_dir):
            os.mkdir(cur_pic_dir)
        if not os.path.isdir(cur_anno_dir):
            os.mkdir(cur_anno_dir)
        if not os.path.isdir(cur_save_dir):
            os.mkdir(cur_save_dir)
        ret, f = cap.read()
        if f is None:
            break
        cv2.imwrite(cur_pic_dir + '%07d.jpeg' % (idx%512), f)
        detect(f, cur_pic_dir + "%07d.jpeg" % (idx%512), anno_path=cur_anno_dir)
        print('+++++++++++++')
        idx += 1
    cap.release()
    return True


def gen_video_dir(dir):
    file_list = glob(os.path.join(dir,'*'))
    print('video files:')
    print(file_list)
    for f in file_list:
        save_dir = os.path.splitext(f)[0]
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if gen(f) == False:
            break

if __name__ == '__main__':
    v_dir = '/disk2/wanyx/test_ssd'
    gen_video_dir(v_dir)

