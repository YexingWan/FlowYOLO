import xml.etree.ElementTree as ET

'''
out: tuple of tuple of (xmax,xmin,ymax,ymin,class_label)
info: dictionary with items <folder:str, filename:str, size:(w,h)>
'''
def write_annotation(info,out):
    root = ET.Element('annotation')
    ET.SubElement(root,'folder').text = info['folder']
    ET.SubElement(root,'filename').text = info['filename']
    size = ET.SubElement(root,'size')
    object = ET.SubElement(root, 'bject')

    ET.SubElement(size,'weidth').text = info['size'][0]
    ET.SubElement(size,'height').text = info['size'][1]

    for box in out:
        ET.SubElement(object,'name').text = box[4]
        budbox = ET.SubElement(object, 'budbox')
        ET.SubElement(budbox, 'xmax').text = box[0]
        ET.SubElement(budbox, 'xmin').text = box[1]
        ET.SubElement(budbox, 'ymax').text = box[2]
        ET.SubElement(budbox, 'ymin').text = box[3]





