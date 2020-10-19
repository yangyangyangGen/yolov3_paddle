from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
sys.path.append("../utils")
from box import iou_bboxes_xywh, crop_bbox_x1y1wh



import numpy as np
import cv2
import random

from PIL import Image, ImageEnhance
from math import sqrt


__all__ = ["random_distort", "random_expand"]


def random_distort(cv_img):
    """Random改变 亮度、对比度、颜色."""

    def random_brightness(im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(im).enhance(e)

    def random_contrast(im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(im).enhance(e)

    def random_color(im, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(im).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
    pil_im = Image.fromarray(cv_img)
    for op in ops:
        pil_im = op(pil_im)
    return np.asarray(pil_im)


def random_expand(cv_img,
                  gt_xywh,
                  max_ratio=4.,
                  fill=None,
                  xy_ratio_same=True,
                  thresh=.5,
                  xywh_is_normalize=True,
                  xywh_do_normalize=True):
    """
    随机填充. 创建一个大的画布进行填充.
    Create a Large Background and do fill.

    :param max_ratio: 最大比例(相对于原图)
    :param fill: 画布背景颜色.
    :param xy_ratio_same: 长宽最大比例是否相同
    :param xywh_is_normalize: gt中的xywh是否归一化
    :param xywh_do_normalize: gt中xywh是否要进行归一化
    :return: image, gt_xywh
    """

    if xywh_is_normalize:
        assert "float" in str(gt_xywh.dtype), \
            "gt_xywh normalized and param `gt_xywh` dtype is not float."

    if random.random() > thresh:
        return cv_img, gt_xywh
    if max_ratio < 1.:  # 画布需要比原来的大.
        return cv_img, gt_xywh

    if xywh_do_normalize:
        gt_xywh = gt_xywh.astype(np.float)

    h, w, c = cv_img.shape
    ratio_x = np.random.uniform(1, max_ratio)
    ratio_y = ratio_x if xy_ratio_same else np.random.uniform(1, max_ratio)

    oh = int(h * ratio_y)
    ow = int(w * ratio_x)

    offset_x = random.randint(0, ow - w)
    offset_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c), dtype=cv_img.dtype)
    print(fill)
    if fill is not None and len(fill) == c:
        for i in range(len(fill)):
            out_img[..., i] = fill[i] * 255.  # todo: 为什么乘255.
            # out_img[..., i] = fill[i]

    out_img[offset_y: offset_y+h, offset_x: offset_x+w, :] = cv_img

    norm_fscale = lambda x, y=1.: float(x) if xywh_do_normalize else float(y)

    if xywh_is_normalize:
        gt_xywh[:, 0] = (gt_xywh[:, 0] * w + offset_x) / norm_fscale(ow)
        gt_xywh[:, 1] = (gt_xywh[:, 1] * h + offset_y) / norm_fscale(oh)

        gt_xywh[:, 2] = gt_xywh[:, 2] / norm_fscale(ratio_x, 1./w)
        gt_xywh[:, 3] = gt_xywh[:, 3] / norm_fscale(ratio_y, 1./h)
    else:
        gt_xywh[:, 0] = (gt_xywh[:, 0] + offset_x) / norm_fscale(ow)
        gt_xywh[:, 1] = (gt_xywh[:, 1] + offset_y) / norm_fscale(oh)

        gt_xywh[:, 2] = gt_xywh[:, 2] / norm_fscale(ow)
        gt_xywh[:, 3] = gt_xywh[:, 3] / norm_fscale(oh)

    return out_img, gt_xywh


def random_crop(cv_img, gt_xywh, gt_clas,
                scales=(0.3, 1.0), max_ratio=2.0,
                constraints=None, max_trial=50):
    """随机裁剪"""

    assert len(scales) == 2

    if len(gt_xywh) == 0:
        print("Warning: gt_xywh length is 0.")
        return cv_img, gt_xywh, gt_clas

    if constraints is None:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]

    pil_img = Image.fromarray(cv_img)
    w, h = pil_img.size

    crop_boxes = [(0, 0, w, h)]
    # crop_boxes = []
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            # generator crop box.
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1. / max_ratio, scale * scale),
                                          min(max_ratio, 1. / scale / scale))
            crop_h = int(h * scale / sqrt(aspect_ratio))
            crop_w = int(w * scale * sqrt(aspect_ratio))
            # crop_x, crop_y of left bottom.
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_xywh_normed = np.array([[(crop_x + crop_w / 2.) / w,
                                          (crop_y + crop_h / 2.) / h,
                                          crop_w / float(w), crop_h / float(h)]], dtype="float")

            iou_ndarray = iou_bboxes_xywh(crop_xywh_normed, gt_xywh)
            if min_iou <= iou_ndarray.min() and max_iou >= iou_ndarray.max():
                crop_boxes.append((crop_x, crop_y, crop_w, crop_h))
                break

    # crop
    print("crop_boxes", crop_boxes)
    while crop_boxes:
        crop_box = crop_boxes.pop(random.randint(0, len(crop_boxes) - 1))
        out_crop_boxes, out_crop_labels, life_box_num = crop_bbox_x1y1wh(gt_xywh, gt_clas, crop_box, (h, w))
        if life_box_num < 1:
            continue

        # why resize. ???
        pil_img = pil_img.crop((crop_box[0], crop_box[1],
                                crop_box[0] + crop_box[2], crop_box[1] + crop_box[3])).resize(pil_img.size, Image.LANCZOS)
        out_img = np.asarray(pil_img)
        return out_img, out_crop_boxes, out_crop_labels

    return cv_img, gt_xywh, gt_clas


if __name__ == "__main__":
    pass


