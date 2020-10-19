"""

data aug test ok function:
        random_expand, random_distort, random_crop, 
        random_interp_zoom, random_hflip, random_vflip



"""

import cv2
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
import random
from matplotlib import pyplot as plt
import sys
sys.path.append("../data")
from aug import random_expand, random_distort, random_crop, random_interp_zoom, random_hflip, random_vflip, image_augment

from annotation import get_cname2cid_dict_from_txt, voc_parse

from wrapper import get_img_data_from_record_dict


# Define global image and gt.
fname = r"D:\workspace\DataSets\det\Insect\JPEGImages\train\1.jpeg"
im = cv2.imread(fname)[..., ::-1]

bboxes_xyxy = [[473, 578, 612, 727],
               [624, 488, 711, 554],
               [756, 786, 841, 856],
               [607, 781, 690, 842],
               [822, 505, 948, 639]]
bboxes_xyxy = np.asarray(bboxes_xyxy)

bboxes_xywh = np.zeros_like(bboxes_xyxy)
bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

h, w = im.shape[:2]
bboxes_xywh_normed = bboxes_xywh.astype("float32")
bboxes_xywh_normed[..., 0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
bboxes_xywh_normed[..., 1::2] = bboxes_xywh_normed[..., 1::2] / float(h)



def draw_rectangle(currentAxis, bbox,
                   edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # bbox_xywh: x y is center point.
    rect = patches.Rectangle((bbox[0] - bbox[2] / 2., bbox[1] - bbox[3] / 2.),
                             bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor,
                             facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)

def xywh_encode(boxes: np.ndarray, hw):
    assert boxes.shape[-1] == 4
    boxes = boxes.copy()
    boxes = boxes.astype("float32")
    boxes[..., 0::2] /= float(hw[1])
    boxes[..., 1::2] /= float(hw[0]) 
    return boxes

def xywh_decode(boxes: np.ndarray, hw):
    assert boxes.shape[-1] == 4
    boxes = boxes.copy()
    boxes[..., 0::2] *= float(hw[1])
    boxes[..., 1::2] *= float(hw[0]) 
    return boxes.astype("int32")


def test_random_distort():
    rows = 3
    cols = 3
    im = cv2.imread(fname)[..., ::-1]

    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows, cols, r * cols + c + 1)
            out = random_distort(im)
            plt.imshow(out)
            plt.axis("off")
    plt.show()

    status = False


def test_random_expand(status=True):
    """
    测试参数以及结果.
        do_norm     is_norm        result    function
            0           0            1          fn1
            0           1            1          fn2
            1           0            1          fn3
            1           1            1          fn4

    fn5: 测试fill参数/
    """

    im = cv2.imread(fname)[..., ::-1]

    def fn1():
        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.
        currentAxis = plt.gca()
        ret_im, ret_bboxes = random_expand(im, bboxes_xywh, thresh=1,
                                           xy_ratio_same=False,
                                           xywh_do_normalize=False, xywh_is_normalize=False)
        plt.imshow(ret_im.astype("uint8"))
        for bbox in ret_bboxes:
            draw_rectangle(currentAxis, bbox, edgecolor="b")

    def fn2():
        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

        h, w = im.shape[:2]
        bboxes_xywh_normed = bboxes_xywh.astype("float32")
        bboxes_xywh_normed[...,
                           0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
        bboxes_xywh_normed[...,
                           1::2] = bboxes_xywh_normed[..., 1::2] / float(h)

        currentAxis = plt.gca()
        ret_im, ret_bboxes = random_expand(im, bboxes_xywh_normed, thresh=1,
                                           xy_ratio_same=False,
                                           xywh_do_normalize=False, xywh_is_normalize=True)
        plt.imshow(ret_im.astype("uint8"))
        for bbox in ret_bboxes:
            draw_rectangle(currentAxis, bbox, edgecolor="b")

    def fn3():

        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

        currentAxis = plt.gca()
        ret_im, ret_bboxes = random_expand(im, bboxes_xywh, thresh=1,
                                           xy_ratio_same=False,
                                           xywh_do_normalize=True, xywh_is_normalize=False)

        oh, ow = ret_im.shape[:2]
        ret_bboxes_decode = ret_bboxes.astype(np.float)
        ret_bboxes_decode[:, 0::2] = ret_bboxes_decode[:, 0::2] * ow
        ret_bboxes_decode[:, 1::2] = ret_bboxes_decode[:, 1::2] * oh
        plt.imshow(ret_im.astype("uint8"))
        for bbox in ret_bboxes_decode:
            draw_rectangle(currentAxis, bbox, edgecolor="b")

    def fn4():

        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

        h, w = im.shape[:2]
        bboxes_xywh_normed = bboxes_xywh.astype("float32")
        bboxes_xywh_normed[...,
                           0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
        bboxes_xywh_normed[...,
                           1::2] = bboxes_xywh_normed[..., 1::2] / float(h)

        currentAxis = plt.gca()
        ret_im, ret_bboxes = random_expand(im, bboxes_xywh_normed, thresh=1,
                                           xy_ratio_same=False,
                                           xywh_do_normalize=True, xywh_is_normalize=True)

        oh, ow = ret_im.shape[:2]
        ret_bboxes_decode = ret_bboxes.astype(np.float)
        ret_bboxes_decode[:, 0::2] = ret_bboxes_decode[:, 0::2] * ow
        ret_bboxes_decode[:, 1::2] = ret_bboxes_decode[:, 1::2] * oh
        plt.imshow(ret_im.astype("uint8"))
        for bbox in ret_bboxes_decode:
            draw_rectangle(currentAxis, bbox, edgecolor="b")

    def fn5():
        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

        h, w = im.shape[:2]
        bboxes_xywh_normed = bboxes_xywh.astype("float32")
        bboxes_xywh_normed[...,
                           0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
        bboxes_xywh_normed[...,
                           1::2] = bboxes_xywh_normed[..., 1::2] / float(h)

        currentAxis = plt.gca()
        ret_im, ret_bboxes = random_expand(im, bboxes_xywh_normed, thresh=1,
                                           xy_ratio_same=False,
                                           # fill=[np.mean(im[..., i]) for i in range(im.shape[-1])],
                                           fill=[100, 100, 100],
                                           xywh_do_normalize=True, xywh_is_normalize=True)

        oh, ow = ret_im.shape[:2]
        ret_bboxes_decode = ret_bboxes.astype(np.float)
        ret_bboxes_decode[:, 0::2] = ret_bboxes_decode[:, 0::2] * ow
        ret_bboxes_decode[:, 1::2] = ret_bboxes_decode[:, 1::2] * oh
        plt.imshow(ret_im.astype("uint8"))
        for bbox in ret_bboxes_decode:
            draw_rectangle(currentAxis, bbox, edgecolor="b")

    if status:
        plt.subplot(221)
        fn1()
        plt.subplot(222)
        fn2()  #
        plt.subplot(223)
        fn3()
        plt.subplot(224)
        fn4()
        plt.show()
    else:
        fn5()
        plt.show()


def test_random_crop():
    im = cv2.imread(fname)
    assert im is not None, f"cv read {fname} return None."
    im = im[..., ::-1]  # bgr -> rgb.

    bboxes_xywh = np.zeros_like(bboxes_xyxy)
    bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
    bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
    bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
    bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.

    num_clas = 10
    fake_clas = np.random.randint(0, num_clas, size=len(bboxes_xyxy, ))

    h, w = im.shape[:2]
    bboxes_xywh_normed = bboxes_xywh.astype("float32")
    bboxes_xywh_normed[..., 0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
    bboxes_xywh_normed[..., 1::2] = bboxes_xywh_normed[..., 1::2] / float(h)

    plt.subplot(121)
    plt.imshow(im.astype("uint8"))
    plt.title("src")
    plt.axis("off")
    currentAxis = plt.gca()
    for xywh in bboxes_xywh:
        draw_rectangle(currentAxis, xywh, edgecolor='b')

    out_img, out_xywh_normed, out_clas = random_crop(
        im, bboxes_xywh_normed, fake_clas)

    plt.subplot(122)

    out_h, out_w = out_img.shape[:2]
    print(out_h, out_w)

    out_xywh = out_xywh_normed.copy()
    out_xywh[..., 0::2] *= out_w
    out_xywh[..., 1::2] *= out_h

    plt.imshow(out_img.astype("uint8"))
    plt.title("dst")
    plt.axis("off")

    print(out_xywh)

    currentAxis = plt.gca()
    for xywh in out_xywh:
        draw_rectangle(currentAxis, xywh, edgecolor='b')

    plt.show()


def test_random_interp_zoom():
    global im, bboxes_xywh_normed, bbox_xywh
    mpl.rcParams["figure.dpi"] = 120
    
    init_size = (320, 320)
    im = cv2.resize(im, init_size)
    
    hw = im.shape[:2]

    rows = 3
    cols = 3
    stride = 32
    exists_i = 1

    plt.subplot(rows, cols, exists_i)
    plt.imshow(im.astype("uint8"))
    plt.title(f"{hw[0]}@{hw[1]}")
    currentAxis = plt.gca()
    
    bboxes_decoded = xywh_decode(bboxes_xywh_normed, hw)
    for xywh in bboxes_decoded:
        draw_rectangle(currentAxis, xywh, edgecolor="b")

    nrows = rows - 1
    ncols = rows + 1
    random_size = 0.
    
    for r in range(nrows):
        for c in range(ncols):
            ratio = r * ncols + c + 1 + exists_i
            plt.subplot(rows, cols, ratio)

            random_size = random.sample(hw, 1)[0] + \
                stride * ratio

            out_im = random_interp_zoom(im, random_size)
            bboxes_decoded = xywh_decode(bboxes_xywh_normed, out_im.shape[:2])
            
            currentAxis = plt.gca()
            for xywh in bboxes_decoded:
                draw_rectangle(currentAxis, xywh, edgecolor="b")
            
            plt.imshow(out_im.astype("uint8"))
            plt.title(f'{random_size}@{random_size}')

    plt.show()


def test_random_flip():
    """
    Test Result Description:

            `in_place`    function    result
                0           fn1         1
                1           fn2         1
    """
    
    
    global im, bboxes_xywh_normed, bbox_xywh
    mpl.rcParams["figure.dpi"] = 150
    hw = im.shape[:2]
    
    plt.subplot(2, 2, 1)
    plt.imshow(im.astype("uint8"))
    plt.title("src")
    currentAxis = plt.gca()
    bboxes_decoded = xywh_decode(bboxes_xywh_normed.copy(), hw)
    for xywh in bboxes_decoded:
        draw_rectangle(currentAxis, xywh, edgecolor="b")
    
    outs_desc = ["h", "v"]
    
    def fn1():
        outs = []
        outs.append(random_hflip(im, bboxes_xywh_normed, 0.))
        outs.append(random_vflip(im, bboxes_xywh_normed, 0.))
        
        for i, out in enumerate(outs):    
            plt.subplot(2, 2, i+1+1)
            (out_im, out_bboxes_normed) = out
            plt.imshow(out_im.astype("uint8"))
            plt.title(outs_desc[i])
            currentAxis = plt.gca()
            bboxes_decoded = xywh_decode(out_bboxes_normed, out_im.shape[:2])
            for xywh in bboxes_decoded:
                draw_rectangle(currentAxis, xywh, edgecolor="b")
            
    def fn2():
        hflip_im = im.copy()
        vflip_im = im.copy()
        hflip_bboxes = bboxes_xywh_normed.copy()
        vflip_bboxes = bboxes_xywh_normed.copy()
        
        random_hflip(hflip_im, hflip_bboxes, 0., in_place=True)
        random_vflip(vflip_im, vflip_bboxes, 0., in_place=True)
        
        outs = [(hflip_im, hflip_bboxes), (vflip_im, vflip_bboxes)]
        
        for i, out in enumerate(outs):    
                plt.subplot(2, 2, i+1+1)
                (out_im, out_bboxes_normed) = out
                plt.imshow(out_im.astype("uint8"))
                plt.title(outs_desc[i])
                currentAxis = plt.gca()
                bboxes_decoded = xywh_decode(out_bboxes_normed, out_im.shape[:2])
                for xywh in bboxes_decoded:
                    draw_rectangle(currentAxis, xywh, edgecolor="b")
    
    # fn1()     
    # fn2()
    
    plt.show()


def test_image_aug():
    
    
    fpath = r"D:\workspace\DataSets\det\Insect\ImageSets\train.txt"
    cname2cid_map = get_cname2cid_dict_from_txt()
    record_list = voc_parse(cname2cid_map, fpath)
    
    
    img, gt_boxes, gt_labels, scales = get_img_data_from_record_dict(record_list[0])
    size = 512
    out_img, out_gt_boxes, out_gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    
    print(out_img.shape, out_gt_boxes.shape, out_gt_labels.shape)


if __name__ == "__main__":
    # test_random_distort()
    # test_random_expand(True)
    # test_random_crop()
    # test_random_interp_zoom()
    # test_random_flip()
    # test_image_aug()
    
    print(f"{__file__} -> {__name__} result Done.")
