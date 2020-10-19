import os
import cv2
import numpy as np


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


def parse_from_record_dict(record_dict,
                           number_gt=50, bbox_do_normalize=True):
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

    assert os.path.exists(im_file_abspath), "{} not exists.".format(im_file_abspath)
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


if __name__ == "__main__":
    from annotation import voc_parse_generator, get_cname2cid_dict_from_txt

    fpath = r"D:\workspace\DataSets\det\Insect\ImageSets\for_script.txt"
    cname2cid_map = get_cname2cid_dict_from_txt()

    record_genator = voc_parse_generator(cname2cid_map, fpath)

    for i, record in enumerate(record_genator):

        im, gt_xywh, gt_clas, scale = parse_from_record_dict(record, -1, bool(i % 2))

        print("\nINFO:\n\timage size is {} \n\tgt_xywh is {} \n\tgt_clas is {} \n\tscale is {}.\n".
              format(im.shape, gt_xywh, gt_clas, scale))

        if (i + 1) == 3:
            break

