from glob import glob
import os

root_path = '/disk2/wanyx/head_sequence_train/'
sequence_list = []
annotations_list = []
videos = glob(os.path.join(root_path, '*'))
for video in videos:
    sequence_list.extend(glob(os.path.join(video, 'Data/*')))
    annotations_list.extend(glob(os.path.join(video, 'Annotations/*')))

for idx in range(len(sequence_list)):
    files = glob('%s/*.*' % sequence_list[idx])
    anns = glob('%s/*.*' % annotations_list[idx])
    if len(files) != len(anns):
        print('Different length in sequence: %s' % sequence_list[idx])

