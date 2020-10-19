import numpy as np


def iou_bboxes_xywh(bbox1: np.ndarray, bbox2: np.ndarray):
    """
        x y is of center point.
        1: N、 N: N、 1:1
    """
    assert bbox1.shape[-1] == 4 and bbox2.shape[-1] == 4

    bbox1_xmin, bbox1_xmax = bbox1[:, 0] - \
        bbox1[:, 2] / 2., bbox1[:, 0] + bbox1[:, 2] / 2.
    bbox1_ymin, bbox1_ymax = bbox1[:, 1] - \
        bbox1[:, 3] / 2., bbox1[:, 1] + bbox1[:, 3] / 2.

    bbox2_xmin, bbox2_xmax = bbox2[:, 0] - \
        bbox2[:, 2] / 2., bbox2[:, 0] + bbox2[:, 2] / 2.
    bbox2_ymin, bbox2_ymax = bbox2[:, 1] - \
        bbox2[:, 3] / 2., bbox2[:, 1] + bbox2[:, 3] / 2.

    bbox1_area = (bbox1_xmax - bbox1_xmin + 1.) * \
        (bbox1_ymax - bbox1_ymin + 1.)
    bbox2_area = (bbox2_xmax - bbox2_xmin + 1.) * \
        (bbox2_ymax - bbox2_ymin + 1.)

    inter_xmin = np.maximum(bbox1_xmin, bbox2_xmin)
    inter_ymin = np.maximum(bbox1_ymin, bbox2_ymin)
    inter_xmax = np.minimum(bbox2_xmax, bbox2_xmax)
    inter_ymax = np.minimum(bbox1_ymax, bbox2_ymax)

    inter_h = (inter_ymax - inter_ymin + 1.)
    inter_w = (inter_xmax - inter_xmin + 1.)
    #
    inter_h = np.clip(inter_h, a_min=0, a_max=None)
    inter_w = np.clip(inter_w, a_min=0, a_max=None)

    inter_area = inter_h * inter_w

    return inter_area / (bbox1_area + bbox2_area - inter_area).astype("float")


def crop_bbox_x1y1wh(boxes_xywh: np.ndarray, labels: np.ndarray, crop_x1y1wh, im_hw):
    """For Data aug.
    删除(置0)不在crop区域的box.
    """

    xmin, ymin, w, h = map(float, crop_x1y1wh)
    im_h, im_w = map(float, im_hw)

    boxes = boxes_xywh.copy()
    # xywh -> xyxy
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2.) * im_w, \
        (boxes[:, 0] + boxes[:, 2] / 2) * im_w

    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2.) * im_h, \
        (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_xyxy = np.array([xmin, ymin, xmin + w, ymin + h])
    boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2.

    # get [N, ] 代表crop的框是否包含了此物体
    mask = np.logical_and(
        crop_xyxy[:2] <= boxes_center, boxes_center <= crop_xyxy[2:]).all(axis=1)

    # align
    boxes[:, :2] = np.maximum(boxes[:, :2], crop_xyxy[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_xyxy[2:])
    # 左上角平移， 改变box坐标
    boxes[:, :2] -= crop_xyxy[:2]
    boxes[:, 2:] -= crop_xyxy[:2]
    # 判断坐标是否正常， 有点鸡肋.
    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))

    # mask: [N,] -> [N, 1]
    boxes = boxes * np.expand_dims(mask.astype("float32"), axis=1)
    labels = labels * mask.astype("float32")

    # xyxy -> xywh
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2. / w, \
                               (boxes[:, 2] - boxes[:, 0]) / w

    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2. / h, \
                               (boxes[:, 3] - boxes[:, 1]) / h
    return boxes, labels, mask.sum()
