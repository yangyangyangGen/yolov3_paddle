
from box import iou_bboxes_xywh
import numpy as np
import sys
sys.path.append("../utils")


if __name__ == "__main__":
    offset = 10
    bbox1 = np.array(
        [[10, 20, 30, 40],
         [20, 30, 40, 50],
         [30, 40, 50, 60]])
    bbox2 = bbox1 + offset
    bbox3 = np.array(
        [[12, 22, 32, 42]])
    iou = iou_bboxes_xywh(bbox1, bbox3)
    print(iou)
