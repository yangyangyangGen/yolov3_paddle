from aug import random_expand, random_distort, random_crop
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import sys
sys.path.append("../data")


def draw_rectangle(currentAxis, bbox,
                   edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # bbox_xywh: x y is center point.
    rect = patches.Rectangle((bbox[0] - bbox[2] / 2., bbox[1] - bbox[3] / 2.),
                             bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor,
                             facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


fname = r"D:\workspace\DataSets\det\Insect\JPEGImages\train\1.jpeg"

bboxes_xyxy = [[473, 578, 612, 727],
               [624, 488, 711, 554],
               [756, 786, 841, 856],
               [607, 781, 690, 842],
               [822, 505, 948, 639]]
bboxes_xyxy = np.asarray(bboxes_xyxy)


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


def test_random_expand():
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

    status = False
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
    assert im is not None
    im = im[..., ::-1]  # bgr -> rgb.

    bboxes_xywh = np.zeros_like(bboxes_xyxy)
    bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0] + 1.
    bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1] + 1.
    bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.
    bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.
    fake_clas = np.random.randint(0, 10, size=len(bboxes_xyxy, ))

    h, w = im.shape[:2]
    bboxes_xywh_normed = bboxes_xywh.astype("float32")
    bboxes_xywh_normed[..., 0::2] = bboxes_xywh_normed[..., 0::2] / float(w)
    bboxes_xywh_normed[..., 1::2] = bboxes_xywh_normed[..., 1::2] / float(h)

    plt.subplot(121)
    plt.imshow(im.astype("uint8"))
    plt.axis("off")
    currentAxis = plt.gca()
    for xywh in bboxes_xywh:
        draw_rectangle(currentAxis, xywh, edgecolor='b')

    out_img, out_xywh, out_clas = random_crop(
        im, bboxes_xywh_normed, fake_clas)

    plt.subplot(122)
    print(out_xywh)
    plt.imshow(out_img.astype("uint8"))
    plt.axis("off")
    currentAxis = plt.gca()
    for xywh in out_xywh:
        draw_rectangle(currentAxis, xywh, edgecolor='b')

    plt.show()


if __name__ == "__main__":
    # test_random_distort()
    # test_random_expand()
    test_random_crop()

    print("test_aug Done.")
