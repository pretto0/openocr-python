from openocr import OpenOCRRec
from utility import get_image_file_list,check_and_read
import cv2

imgpath = "img/word_1.jpg"
valid_image_file_list = []
img_list = []
text_recognizer = OpenOCRRec()

print(
        "rec_image_shape parameter defaults to '3, 48, 320'"
    )

image_file_list = get_image_file_list(imgpath)
for image_file in image_file_list:
    img, flag, _ = check_and_read(image_file)
    if not flag:
        img = cv2.imread(image_file)
    if img is None:
        print("error in loading image:{}\n".format(image_file))
        continue
    valid_image_file_list.append(image_file)
    img_list.append(img)



rec_res, _ = text_recognizer(img_list)


print(rec_res)

for ino in range(len(img_list)):
    print(
        "Predicts of {}:{}\n".format(valid_image_file_list[ino], rec_res[ino])
    )