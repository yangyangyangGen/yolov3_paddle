import math
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np


def generator_anchor_by_point(x_center, y_center, hw, base_length, scales, ratios):

    bboxes = []
    for scale in scales:
        for ratio in ratios:
            h = base_length * scale * math.sqrt(ratio)
            w = base_length * scale / math.sqrt(ratio)
            x1 = max(x_center - w/2, 0)
            y1 = max(y_center - h/2, 0)
            x2 = min(x_center + w/2 - 1., hw[1] - 1.)
            y2 = min(y_center + h/2 - 1., hw[0] - 1.)
            bboxes.append([x1, y1, x2, y2])

    return bboxes


def box_iou_xyxy(box1, box2):
    # assert

    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    box1_area = np.maximum((y1max - y1min + 1.) * (x1max - x1min + 1.), 0)

    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    box2_area = np.maximum((y2max - y2min + 1.) * (x2max - x2min + 1.), 0)

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)

    inter_h = np.maximum(ymax - ymin + 1., 0)
    inter_w = np.maximum(xmax - xmin + 1., 0)
    intersection = inter_w * inter_h

    union = box1_area + box2_area - intersection

    return float(intersection) / union


def box_iou_xywh(box1, box2):
    # assert

    x1min, y1min = box1[0] - box1[2] / 2., box1[1] - box1[2] / 2.
    x1max, y1max = box1[0] + box1[2] / 2., box1[1] + box1[2] / 2.

    box1_area = np.maximum((y1max - y1min + 1.) * (x1max - x1min + 1.), 0)

    x2min, y2min = box2[0] - box2[2] / 2., box2[1] - box2[2] / 2.
    x2max, y2max = box2[0] + box2[2] / 2., box2[1] + box2[2] / 2.
    box2_area = np.maximum((y2max - y2min + 1.) * (x2max - x2min + 1.), 0)

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)

    inter_h = np.maximum(ymax - ymin + 1., 0)
    inter_w = np.maximum(xmax - xmin + 1., 0)
    intersection = inter_w * inter_h

    union = box1_area + box2_area - intersection

    return float(intersection) / union


if __name__ == "__main__":

    """
    from draw import draw_rectangle
    plt.figure(figsize=(10, 10))
    filename = r"D:\workspace\dl\scratch\paddlepaddle\dygraph\det\yolov3\test\images\test.jpg"
    im = imread(filename)
    plt.imshow(im)

    bboxes = generator_anchor_by_point(500, 500, [3034, 1586], 100, [1., 2.], [.5, 1., 2.])
    currentAxis = plt.gca()

    draw_rectangle(currentAxis, [500, 500, 800, 800], edgecolor='r')

    for bbox in bboxes:
        draw_rectangle(currentAxis, bbox, edgecolor='b')
    plt.show()
    
    """

    """
    bbox1 = [100., 100., 200., 200.]
    bbox2 = [120., 120., 220., 220.]
    iou = box_iou_xyxy(bbox1, bbox2)
    print('box_iou_xyxy IoU is {}'.format(iou))

    bbox1 = [150., 150., 100., 100.]
    bbox2 = [170., 170., 100., 100.]
    iou = box_iou_xywh(bbox1, bbox2)
    print('box_iou_xywh IoU is {}'.format(iou))
    """


    #




