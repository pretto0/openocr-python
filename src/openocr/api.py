from openocr import OpenOCRE2E
from utility import get_image_file_list,check_and_read,draw_ocr_box_txt
import os
import time
import cv2
import numpy as np
import json
from PIL import Image

def infer(imgpath,savepath='./inference_results'):

    image_file_list = get_image_file_list(imgpath)

    text_sys = OpenOCRE2E()

    current_dir = os.path.dirname(__file__)
    font_path = os.path.join(current_dir,"doc/fonts/simfang.ttf")
    drop_score = 0.5
    draw_img_save_dir = savepath 
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []
    print("rec_image_shape parameter defaults to '3, 48, 320'")

    total_time = 0
    _st = time.time()

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
            page_num = 0
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
            

    print("The predict total time is {}\n".format(time.time() - _st))

    with open(
        os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(save_results)