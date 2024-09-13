import sys
import os

import numpy as np
import cv2

import paddle
from paddle import inference

def create_predictor(mode):
    if mode == "det":
        model_dir = "openatom_det_repsvtr_ch_infer/"
    elif mode == "rec":
        model_dir = "openatom_rec_svtrv2_ch_infer/"
    else:
        print("not find {} model".format(mode))
        sys.exit(0)
    
    file_names = ["model", "inference"]

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