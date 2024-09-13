from openocr import OpenOCRDet
from utility import get_image_file_list,check_and_read
import cv2

imgpath = "img/00018069.jpg"

image_file_list = get_image_file_list(imgpath)

text_detector = OpenOCRDet()

save_results = []

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

        dt_boxes, _ = text_detector(img)
        print(dt_boxes)