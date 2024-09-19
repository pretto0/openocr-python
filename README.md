# OpenOCR

This is the application deployment from the [OpenOCR](https://github.com/Topdu/OpenOCR) project. We currently offer three functionalities: text detection, text recognition, and end-to-end recognition. The models are based on the results from the [FVL](https://fvl.fudan.edu.cn)OCR team's performance in the recent [PaddleOCR Algorithm Model Challenge—Task 1: End-to-End OCR Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction). In terms of results, the end-to-end recognition accuracy on the B leaderboard has improved by 2.5% compared to PP-OCRv4, while the inference speed remains the same.

### Installation and Usage

We provide a quick method for deploying OCR inference:

`pip install paddlepaddle-gpu`

`pip install openocr-python`

You can quickly access the features of OpenOCR by using:

`import openocr`

### Quick Inference

Use `openocr.infer(ImgPath)` for fast end-to-end inference on the image at the specified path.

### Features List

OpenOCR currently includes three core inference interfaces, implemented using the class's __call__ method:

- Text Detection
  Use the `OpenOCRDet` class to create a text detector:
  `text_detector = openocr.OpenOCRDet()`
  Then, use `text_detector(img)` to detect text in the image. This text detector returns a list of text bounding boxes found in the image.
- Text Recognition
  Use the `OpenOCRRec` class to create a text recognizer:
  `text_recognizer = openocr.OpenOCRRec()`
  Then, use `text_recognizer(imglist)` to recognize text in the images. The text recognizer accepts a list of image elements and returns the recognition results and inference time in list format.
- End-to-End
  Use the `OpenOCRE2E` class to create an end-to-end recognizer:
  `text_sys = OpenOCRE2E()`
  Then, use `text_sys(img)` to perform detection on the image. The end-to-end recognizer returns a list containing both the detection boxes and the corresponding recognition results.

### Introduction to OpenOCR

OpenOCR aims to establish a unified training and evaluation benchmark for scene text detection and recognition algorithms, at the same time, serves as the official code repository for the OCR team from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University.

We sincerely welcome the researcher to recommend OCR or relevant algorithms and point out any potential factual errors or bugs. Upon receiving the suggestions, we will promptly evaluate and critically reproduce them. We look forward to collaborating with you to advance the development of OpenOCR and continuously contribute to the OCR community!

### Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Thanks for their awesome work!
