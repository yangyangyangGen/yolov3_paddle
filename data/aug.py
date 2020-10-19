"""
    todo:
        1. Use class to overrides.
        2. each function add `in_place` param ????
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from math import sqrt
from PIL import Image, ImageEnhance
import random
import cv2
import numpy as np

import sys
sys.path.append("../utils")
from box import iou_bboxes_xywh, crop_bbox_x1y1wh



# __all__ = ["image_augment"]


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
                  fill_relative=None,
                  xy_ratio_same=True,
                  thresh=.5,
                  xywh_is_normalize=True,
                  xywh_do_normalize=True):
    """
    随机填充. 对原图的外围进行填充 fill, 如果fill为None则填充0.
    Create a Large Background and do fill.

    :param max_ratio: 最大比例(相对于原图)
    :param fill_relative: 填充的像素值.
    :param xy_ratio_same: 长宽最大比例是否相同
    :param xywh_is_normalize: gt中的xywh是否归一化
    :param xywh_do_normalize: gt中xywh是否要进行归一化
    :return: image, gt_xywh
    """

    # TODO: Will do 代码整理.
    assert xywh_is_normalize
    assert xywh_do_normalize

    if xywh_is_normalize:
        assert "float" in str(gt_xywh.dtype), \
            "gt_xywh normalized and param `gt_xywh` dtype is not float."

    if random.random() > thresh:
        return cv_img, gt_xywh
    if max_ratio < 1.:  # 填充后的画布要比原来大.
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

    if fill_relative is not None and len(fill_relative) == c:
        for i in range(len(fill_relative)):
            out_img[..., i] = fill_relative[i] * 255.

    out_img[offset_y: offset_y+h, offset_x: offset_x+w, :] = cv_img

    def norm_fscale(x, y=1.): return float(
        x) if xywh_do_normalize else float(y)

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
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0),
                       (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]

    pil_img = Image.fromarray(cv_img)
    w, h = pil_img.size

    # crop_boxes = [(0, 0, w, h)]
    crop_boxes = []
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
    while crop_boxes:
        crop_box = crop_boxes.pop(random.randint(0, len(crop_boxes) - 1))
        out_crop_boxes, out_crop_labels, life_box_num = crop_bbox_x1y1wh(
            gt_xywh, gt_clas, crop_box, (h, w))

        if life_box_num < 1:
            continue

        # at this bbox is relative coord.
        # bbox是相对坐标.
        pil_img = pil_img.crop((crop_box[0], crop_box[1],
                                crop_box[0] + crop_box[2], crop_box[1] + crop_box[3])).resize(pil_img.size, Image.LANCZOS)
        out_img = np.asarray(pil_img)

        print("will return.")

        return out_img, out_crop_boxes, out_crop_labels

    return cv_img, gt_xywh, gt_clas


def random_interp_zoom(cv_img, size,
                       interp_method_tuple=(cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                            cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)):
    assert interp_method_tuple
    inter_method = interp_method_tuple[random.randint(
        0, len(interp_method_tuple) - 1)]

    h, w = cv_img.shape[:2]

    scale_x = size / float(w)
    scale_y = size / float(h)

    out_img = cv2.resize(cv_img, None, None, scale_x,
                         scale_y, interpolation=inter_method)
    return out_img


def random_hflip(cv_img, gt_xywh, thresh=.5, in_place=False):
    if random.random() > thresh:
        if in_place:
            cv_img[...] = cv_img[:, ::-1, :]
            gt_xywh[:, 0] = 1. - gt_xywh[:, 0]
            return
        else:
            gt_xywh = gt_xywh.copy()
            cv_img = cv_img[:, ::-1, :]
            gt_xywh[:, 0] = 1. - gt_xywh[:, 0]
    return cv_img, gt_xywh


def random_vflip(cv_img, gt_xywh, thresh=.5, in_place=False):
    if random.random() > thresh:
        if in_place:
            cv_img[...] = cv_img[::-1, ::, :]
            gt_xywh[:, 1] = 1. - gt_xywh[:, 1]
            return
        else:
            gt_xywh = gt_xywh.copy()
            cv_img = cv_img[::-1, ::, :]
            gt_xywh[:, 1] = 1. - gt_xywh[:, 1]
    return cv_img, gt_xywh


def random_shuffle_boxes(gt_boxes, gt_clas):
    assert len(gt_boxes) == len(gt_clas)
    ridx = np.random.permutation(len(gt_boxes))
    return gt_boxes[ridx], gt_clas[ridx]


def image_augment(cv_img, gt_xywh: np.ndarray, gt_clas: np.ndarray, size: int,
                  means=None, once_crop=True):
    cv_img = random_distort(cv_img)
    cv_img, gt_xywh = random_expand(cv_img, gt_xywh)
    cv_img, gt_xywh, gt_clas = random_crop(cv_img, gt_xywh, gt_clas)
    cv_img = random_interp_zoom(cv_img, size)

    '''
    cv_img = random_hflip(cv_img, gt_xywh)
    if not once_crop:
        cv_img = random_hflip(cv_img, gt_xywh)
    '''

    if once_crop:
        cv_img, gt_xywh = random_hflip(cv_img, gt_xywh) if random.randint(
            0, 1) else random_vflip(cv_img, gt_xywh)
    else:
        if random.randint(0, 1):
            cv_img, gt_xywh = random_hflip(cv_img, gt_xywh)
            cv_img, gt_xywh = random_vflip(cv_img, gt_xywh)
        else:
            cv_img, gt_xywh = random_vflip(cv_img, gt_xywh)
            cv_img, gt_xywh = random_hflip(cv_img, gt_xywh)

    gt_xywh, gt_clas = random_shuffle_boxes(gt_xywh, gt_clas)

    return cv_img.astype("float32"), gt_xywh.astype("float32"), gt_clas.astype("int32")


if __name__ == "__main__":
    pass
