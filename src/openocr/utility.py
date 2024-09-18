import sys
import os

import numpy as np
import cv2

import random
import math
import PIL
from PIL import Image, ImageDraw, ImageFont

import paddle
from paddle import inference
from check_file import ensure_files_exist

def create_predictor(mode):
    current_dir = os.path.dirname(__file__)

    if mode == "det":
        model_dir = os.path.join(current_dir,"openatom_det_repsvtr_ch_infer/")
        url = 'https://paddleocr.bj.bcebos.com/openatom/openatom_det_repsvtr_ch_infer.tar'
        tar_file_path = os.path.join(current_dir,"openatom_det_repsvtr_ch_infer.tar")
    elif mode == "rec":
        model_dir = os.path.join(current_dir,"openatom_rec_svtrv2_ch_infer/")
        url = 'https://paddleocr.bj.bcebos.com/openatom/openatom_rec_svtrv2_ch_infer.tar'
        tar_file_path = os.path.join(current_dir,"openatom_rec_svtrv2_ch_infer.tar")
    else:
        print("not find {} model".format(mode))
        sys.exit(0)
    
    file_names = ["model", "inference"]

    ensure_files_exist(model_dir,tar_file_path,url)

    for file_name in file_names:
        model_file_path = "{}/{}.pdmodel".format(model_dir, file_name)
        params_file_path = "{}/{}.pdiparams".format(model_dir, file_name)
        if os.path.exists(model_file_path) and os.path.exists(params_file_path):
            break
    if not os.path.exists(model_file_path):
        raise ValueError(
            "not find model.pdmodel or inference.pdmodel in {}".format(model_dir)
        )
    if not os.path.exists(params_file_path):
        raise ValueError(
            "not find model.pdiparams or inference.pdiparams in {}".format(
                model_dir
            )
        )

    config = inference.Config(model_file_path, params_file_path)

    precision = inference.PrecisionType.Float32

    gpu_id = get_infer_gpuid()

    if gpu_id is None:
        print(
            "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
        )
        config.disable_gpu()
    else:
        config.enable_use_gpu(500, 0)

    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)


    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()

    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(mode, predictor)
    return predictor, input_tensor, output_tensors, config

def get_infer_gpuid():
    """
    Get the GPU ID to be used for inference.

    Returns:
        int: The GPU ID to be used for inference.
    """
    # logger = get_logger()
    if not paddle.device.is_compiled_with_rocm:
        gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        gpu_id_str = os.environ.get("HIP_VISIBLE_DEVICES", "0")

    gpu_ids = gpu_id_str.split(",")
    # logger.warning(
    print(
        "The first GPU is used for inference by default, GPU ID: {}".format(gpu_ids[0])
    )
    return int(gpu_ids[0])

def get_output_tensors(mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec":
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        print(img_file)
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == "gif":
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print("Cannot read. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        from paddle.utils import try_import

        fitz = try_import("fitz")
        from PIL import Image

        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=None,
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)

def draw_text_det_res(dt_boxes, img):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    return img

def draw_box_txt_fine(img_size, box, txt, font_path=None):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text

def create_font(txt, sz, font_path=None):
    # font_size = int(sz[1] * 0.99)
    from PIL import ImageFont
    font = ImageFont.load_default()
    # if int(PIL.__version__.split(".")[0]) < 10:
    #     length = font.getsize(txt)[0]
    # else:
    #     length = font.getlength(txt)

    # if length > sz[0]:
    #     font_size = int(font_size * sz[0] / length)
    #     font = ImageFont.load_default()
    return font