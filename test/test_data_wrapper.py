"""
"""

import cv2
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
import random
from matplotlib import pyplot as plt

import sys
sys.path.append("../data")

from aug import image_augment
from annotation import get_cname2cid_dict_from_txt, voc_parse
from wrapper import get_img_data_from_record


def test_image_aug():
    """Can not call."""
    fpath = r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt"
    cname2cid_map = get_cname2cid_dict_from_txt()
    record_list = voc_parse(cname2cid_map, fpath)
    
    img, gt_boxes, gt_labels, scales = get_img_data_from_record_dict(record_list[0])
    size = 512
    out_img, out_gt_boxes, out_gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    print(out_img.shape, out_gt_boxes.shape, out_gt_labels.shape)

def test_data_wrapper():
    fpath = r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt"
    cname2cid_map = get_cname2cid_dict_from_txt()
    record_list = voc_parse(cname2cid_map, fpath)
    img, gt_boxes, gt_labels, hw = get_img_data_from_record(record_list[0])
    print(img.shape, gt_boxes.shape, gt_labels.shape, hw)

if __name__ == "__main__":
    # test_image_aug()  # desc
    test_data_wrapper()
    print(f"Task[Test]: {__file__} -> {__name__} result Done.")
