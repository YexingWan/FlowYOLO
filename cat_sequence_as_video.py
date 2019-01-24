import cv2
from glob import glob
import numpy as np
import os

def write_frame(frame, v_writer):
    if v_writer is not None:
        if v_writer.isOpened():
            # Now we can save it to a numpy array.
            v_writer.write(frame)
        else:
            v_writer = cv2.VideoWriter("output/result.avi",
                                       apiPreference=cv2.CAP_ANY,
                                       fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                       fps=int(15),
                                       frameSize=(frame.shape[1], frame.shape[0]))
            # print("data_shape:{}".format(img.shape))
            v_writer.write(frame)
    return v_writer


def gen_video(dir1,dir2):

    seq1 = glob(os.path.join(dir1, '*')).sorted()
    seq2 = glob(os.path.join(dir2, '*')).sorted()

    assert(len(seq1)==len(seq2))

    v_writer = None
    for img1, img2 in zip(seq1,seq2):
        h1 ,w1 = img1.shape
        h2 ,w2 = img2.shape

        assert(h1 < h2 and w1 < w2)


        d_w2 = h2 / h1 * w1
        if d_w2 >= w2:
            dim_diff = np.abs(d_w2 - w2)
        else:
            dim_diff = np.abs((w2 / w1 * h1) - h2)

        # dim_diff = np.abs(h2 - w2)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if d_w2 < w2 else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding (127.5)
        img2 = np.pad(img2, pad, 'constant', constant_values=127.5)
        # Resize
        img2 = cv2.resize(img2, img1.shape)
        frame = np.concatenate([img1,img2],axis=1)
        v_writer = write_frame(frame,v_writer)
    if v_writer is not None:
        v_writer.release()


if __name__=='__main__':
    dir1 = "/home/user/wanyx/FlowYOLO/compared/"
    dir2 = "/home/user/wanyx/FlowYOLO/compared/"
    gen_video(dir1,dir2)