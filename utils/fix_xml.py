from glob import glob
import xml.etree.ElementTree as ET
import os

path = "/disk2/wanyx/head_sequence_train/"

for v in glob(os.path.join(path, '*')):
    for xml_file in glob(os.path.join(v, "Annotations/**/*.xml"), recursive=True):
        print('modifying %s' % xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for box in tree.iter("budbox"):
            xmin = box.find("xmin").text
            ymin = box.find("ymin").text
            xmax = box.find("xmax").text
            ymax = box.find("ymax").text
            boxes.append((xmin,ymin,xmax,ymax))

        root.remove(root.find('object'))
        for box in boxes:
            obj = ET.SubElement(root,'object')
            ET.SubElement(obj,'name').text = 'head'
            bndbox = ET.SubElement(obj,'bndbox')
            ET.SubElement(obj, 'xmin').text = box[0]
            ET.SubElement(obj, 'ymin').text = box[1]
            ET.SubElement(obj, 'xmax').text = box[2]
            ET.SubElement(obj, 'ymax').text = box[3]
        tree.write(xml_file)

