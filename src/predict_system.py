import os
import sys
import subprocess

from utility import get_image_file_list
import logging
import cv2
import copy
import numpy as np
import json
import time
# import logging
from PIL import Image
# from ppocr.utils.logging import get_logger

from utility import parse_args
from utility import slice_generator, merge_fragmented
from utility import get_rotate_crop_image, get_minarea_rect_crop
from utility import check_and_read, draw_ocr_box_txt


from predict_det import TextDetector
from predict_rec import TextRecognizer

class TextSystem(object):
    def __init__(self, args):
        # if not args.show_log:
        #     logger.setLevel(logging.INFO)

        self.text_detector = TextDetector(args)
        self.text_recognizer = TextRecognizer(args)
        self.drop_score = args.drop_score

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
            print(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True, slice={}):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            print("no valid image provided.\n")
            # logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()

        #slice 如是，将图片按照给定比例切片再输送给检测模型

        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
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
        #对检测框进行排序
        dt_boxes = sorted_boxes(dt_boxes)
        #根据检测框对图片进行裁剪
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        
        if len(img_crop_list) > 1000:
            print(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        print("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
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



def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id :: args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    print(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320\n"
    )

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                print("error in loading image:{}\n".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            if len(imgs) > 1:
                print(
                    str(idx)
                    + "_"
                    + str(index)
                    + "  Predict time of %s: %.3fs" % (image_file, elapse)
                    +"\n"
                )
            else:
                print(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse)
                    +"\n"
                )
            for text, score in rec_res:
                print("{}, {:.3f}\n".format(text, score))

            res = [
                {
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                }
                for i in range(len(dt_boxes))
            ]
            if len(imgs) > 1:
                save_pred = (
                    os.path.basename(image_file)
                    + "_"
                    + str(index)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            else:
                save_pred = (
                    os.path.basename(image_file)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path,
                )
                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                    draw_img[:, :, ::-1],
                )
                print(
                    "The visualized image saved in {}\n".format(
                        os.path.join(draw_img_save_dir, os.path.basename(save_file))
                    )
                )

    print("The predict total time is {}\n".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    with open(
        os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(save_results)


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = (
                [sys.executable, "-u"]
                + sys.argv
                + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
            )
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)