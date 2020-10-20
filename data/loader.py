from .wrapper import get_cname2cid_dict_from_txt, make_ndarray
from .annotation import voc_parse


def get_data_loader(label_list_txt: str, image_label_path_mapper_txt: str,
                    batch_size: int, is_train: int, drop_last=False):
    """
    singleProcess.
    @param: is_train:
                > 0 train mode,
                = 0 valid mode,
                < 0 test_mode.
    """

    assert os.path.exists(label_list_txt), \
        f"{label_list_txt} not exists."
    assert os.path.exists(image_label_path_mapper_txt), \
        f"{image_label_path_mapper_txt} not exists."

    cname2cid_map = get_cname2cid_dict_from_txt(label_list_txt)

    records = voc_parse(cname2cid_map, image_label_path_mapper_txt)

    def reader():
        idx = list(range(len(records)))
        if is_train > 0:
            random.shuffle(idx)

        batch = []
        im_size = get_image_size(is_train)
        for i in idx:
            batch.append(get_img_data_from_record(records[i], im_size))
            if len(batch) == batch_size:
                yield make_ndarray(batch)
                batch = []
                im_size = get_image_size(is_train)

        if not drop_last and len(batch):
            yield make_ndarray(batch)

    return reader


def get_multithread_data_loader(label_list_txt: str, image_label_path_mapper_txt: str,
                                batch_size: int, is_train: int, num_thread: int, buffer_size: int, drop_last=False):
    """
    By Paddle Implemention Multi thread data loader.
    @param: is_train:
                > 0 train mode,
                = 0 valid mode,
                < 0 test_mode.
    """
    import functools
    import paddle

    assert os.path.exists(label_list_txt), \
        f"{label_list_txt} not exists."
    assert os.path.exists(image_label_path_mapper_txt), \
        f"{image_label_path_mapper_txt} not exists."

    cname2cid_map = get_cname2cid_dict_from_txt(label_list_txt)

    records = voc_parse(cname2cid_map, image_label_path_mapper_txt)[:1000]

    def reader():
        idx = list(range(len(records)))
        if is_train > 0:
            random.shuffle(idx)

        batch = []
        im_size = get_image_size(is_train)
        for i in idx:
            batch.append((records[i], im_size))
            if len(batch) == batch_size:
                yield batch
                batch = []
                im_size = get_image_size(is_train)

        if not drop_last and len(batch):
            yield batch

    def get_data_fn(items):
        return make_ndarray([get_img_data_from_record(*item)
                             for item in items])
    '''
    def get_data_fn(items):
        batch = []
        for item in items:
            batch.append(get_img_data_from_record(item[0], item[1]))
        return make_ndarray(batch)
    '''
    # mapper = functools.partial(get_data_fn, )
    return paddle.reader.xmap_readers(get_data_fn, reader, num_thread, buffer_size)


​


def test_data_loader(datadir,
                     batch_size=10, test_image_size=608
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """

    image_names = os.listdir(datadir)

    def reader():
        batch_data = []
        img_size = test_image_size

        for image_name in image_names:
            file_path = os.path.join(datadir, image_name)
            assert os.path.exists(file_path), f"{file_path} not exists."
            img = cv2.imread(file_path)[..., ::-1]
            H, W = img.shape[:2]
            img = cv2.resize(img, (img_size, img_size),
                             interpolation=cv2.INTER_LINEAR)
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            chw_img = out_img.astype('float32').transpose((2, 0, 1))
            batch_data.append((image_name.split('.')[0], chw_img, (h, w)))

            if len(batch_data) == batch_size:
                yield make_ndarray(batch_data, -1)
                batch_data = []

        if len(batch_data) > 0:
            yield make_ndarray(batch_data, -1)

    return reader
