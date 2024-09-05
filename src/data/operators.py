import numpy as np


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.keep_ratio = False
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

    def __call__(self, data):
        img = data["image"]
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data["image"] = img
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data
    
class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data
    

class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data

class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list