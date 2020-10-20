import os
import cv2
import numpy as np
import random

from .aug import image_augment
# from aug import image_augment


def get_bbox_N(gt_bbox, gt_clas, N):
    assert N != 0
    assert len(gt_bbox) == len(gt_clas)
    assert isinstance(gt_bbox, np.ndarray)
    assert isinstance(gt_clas, np.ndarray)
    if N == -1:
        return gt_bbox, gt_clas

    if len(gt_clas) > N:
        return gt_bbox[:N], gt_clas[:N]

    ret_bbox = np.zeros((N, 4), dtype=gt_bbox.dtype)
    ret_clas = np.zeros((N, ),  dtype=gt_clas.dtype)
    ret_bbox[:len(gt_bbox)] = gt_bbox[:]
    ret_clas[:len(gt_clas)] = gt_clas[:]

    return ret_bbox, ret_clas


def parse_record_dict(record_dict,
                      number_gt=50, bbox_do_normalize=True) -> tuple:
    """
        record_desc = {
            "im_file": ,
            "im_id": ,
            "im_h": ,
            "im_w": ,
            "im_d": ,
            "is_crowd": ,
            "difficult": ,
            "gt_class": ,
            "gt_bbox_xywh": ,
            "gt_poly": ,
        }
    """

    im_file_abspath = record_dict["im_file"]
    h = record_dict["im_h"]
    w = record_dict["im_w"]

    gt_xywh = record_dict["gt_bbox_xywh"]
    gt_class = record_dict["gt_class"]

    assert os.path.exists(
        im_file_abspath), "{} not exists.".format(im_file_abspath)
    im = cv2.imread(im_file_abspath)[..., ::-1]  # bgr->rgb.

    # check if h and w in record equals that read from img

    assert im.shape[0] == int(h), "image height of {} inconsistent in record({}) and img file({})".format(
        im_file_abspath, h, im.shape[0])

    assert im.shape[1] == int(w), "image width of {} inconsistent in record({}) and img file({})".format(
        im_file_abspath, w, im.shape[1])

    gt_xywh, gt_class = get_bbox_N(gt_xywh, gt_class, number_gt)

    if bbox_do_normalize:
        # normalize by height or width.
        gt_xywh[:, 0::2] /= float(w)
        gt_xywh[:, 1::2] /= float(h)

    return (im,
            gt_xywh, gt_class,
            (h, w))

# TODO: 使用语法糖实现hook形式api. ???


def get_img_data_from_record(record, size,
                             mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)) -> tuple:

    cv_img, gt_xywh, gt_clas, im_hw = parse_record_dict(record)
    cv_img, gt_xywh, gt_clas = image_augment(cv_img, gt_xywh, gt_clas, size)

    mean = np.array(mean, dtype="float32").reshape((1, 1, -1))
    std = np.array(std, dtype="float32").reshape((1, 1, -1))
    #
    cv_img = (cv_img / 255. - mean) / std
    chw_img = cv_img.astype("float32").transpose((2, 0, 1))

    return (chw_img, gt_xywh, gt_clas, im_hw)


def get_image_size(is_train: int, max_stride=32) -> int:
    """
    @param: is_train:
                > 0 train mode,
                = 0 valid mode,
                < 0 test_mode.

    Train or Valid: Multi scale: random ~[320, 608, 32]
    Test: 608
    """
    size = 0
    if is_train >= 0:
        size = 320 + max_stride * random.choice(range(10))
    else:
        size = 608
    return size


def make_ndarray(batch_data, is_train=1):
    if is_train >= 0:
        im_nd = np.array([item[0]
                          for item in batch_data], dtype="float32")
        xywh_nd = np.array([item[1]
                            for item in batch_data], dtype="float32")
        clas_nd = np.array([item[2]
                            for item in batch_data], dtype="int32")
        hw_nd = np.array([item[3]
                          for item in batch_data], dtype="int32")
        return im_nd, xywh_nd, clas_nd, hw_nd
    else:
        img_name_array = np.array([item[0]
                                   for item in batch_data])
        img_data_array = np.array([item[1]
                                   for item in batch_data], dtype='float32')
        img_scale_array = np.array([item[2]
                                    for item in batch_data], dtype='int32')
        return img_name_array, img_data_array, img_scale_array


if __name__ == "__main__":
    '''
    data_loader = get_data_loader(r"D:\workspace\DataSets\det\Insect\ImageSets\label_list.txt",
                                  r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt",
                                  8, 1)

    for batch_im, batch_bbox, batch_clas, batch_hw in data_loader():
        print(
            f"{batch_im.shape} {batch_bbox.shape} {batch_clas.shape} {batch_hw.shape}")
    '''

    """"
    from functools import wraps
    import time
    # time hook.
    def timing(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.clock()
            r = func(*args, **kwargs)
            end = time.clock()
            print('[' + func.__name__ + ']used:' + str(end - start))
            return r
        return wrapper

    @timing
    def fn():
        multithread_data_loader = get_multithread_data_loader(
            r"D:\workspace\DataSets\det\Insect\ImageSets\label_list.txt",
            r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt",
            8, 1, num_thread=1, buffer_size=1)

        for batch_im, batch_bbox, batch_clas, batch_hw in multithread_data_loader():
            print(
                f"{batch_im.shape} {batch_bbox.shape} {batch_clas.shape} {batch_hw.shape}")

    fn()
    """"
