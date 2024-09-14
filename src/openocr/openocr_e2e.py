from openocr_det import OpenOCRDet
from openocr_rec import OpenOCRRec

import time
import copy
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

from utility import get_rotate_crop_image

class OpenOCRE2E(object):
    def __init__(self):
        self.text_detector = OpenOCRDet()
        self.text_recognizer = OpenOCRRec()
        self.drop_score = 0.5
        self.crop_image_res_index = 0

    def __call__(self, img, cls=True):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            print("no valid image provided.\n")
            return None, None, time_dict
        
        start = time.time()
        ori_im = img.copy()

        dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            print("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            print(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )
        
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)
        
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])

            img_crop = get_rotate_crop_image(ori_im, tmp_box)

            img_crop_list.append(img_crop)


        if len(img_crop_list) > 1000:
            print(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )
            
        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        print("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes