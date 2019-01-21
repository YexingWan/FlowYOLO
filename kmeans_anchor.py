import glob
import xml.etree.ElementTree as ET
import numpy as np
from utils.kmeans import kmeans, avg_iou
import os

ANNOTATIONS_PATH = "/disk2/wanyx/head_sequence_train/"
CLUSTERS = 9

def load_dataset(path):
    dataset = []
    for v in glob.glob(os.path.join(path,'*')):
        print(v)
        for xml_file in glob.glob(os.path.join(v,"Annotation/**/*.xml"),recursive=True):
            tree = ET.parse(xml_file)
            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bndbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height
                dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)

print("loading data...")
data = load_dataset(ANNOTATIONS_PATH)
print("start k-means...")
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))
ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))