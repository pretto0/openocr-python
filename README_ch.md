[English](README_en.md) | 简体中文

# OpenOCR

这是来自[OpenOCR](https://github.com/Topdu/OpenOCR)项目的应用端部署，我们目前开放了文本检测，文本识别和端到端三个功能，模型采用[FVL](https://fvl.fudan.edu.cn)OCR团队在最近比赛[PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)的竞赛结果，从效果上看：B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。

### 安装和使用

我们提供一个快速实现OCR推理的部署方法:

`pip install paddlepaddle-gpu`

`pip install openocr-python`

你可以使用
`import openocr`来快速访问openocr的功能。

### 快速推理

使用`openocr.infer(ImgPath)`对所选路径的图片进行快速的端到端推理。

### 功能列表

openocr目前包含三个核心的推理接口，目前的推理接口使用类的`__call__`函数方法实现：

- 文本检测
  文本检测使用OpenOCRDet类，创建一个文本检测器`text_detector = openocr.OpenOCRDet()`并使用`text_detector(img)`对图片进行检测，这个文本检测器会以list的形式返回图片中的文本边界框。

- 文本识别
  文本识别使用OpenOCRRec类，创建一个文本识别器`text_recognizer = openocr.OpenOCRRec()`并使用`text_recognizer(imglist)`对图片进行检测，文本识别器接收img元素的list，返回list格式的识别结果和推理时间。

- 端到端
  端到端识别使用OpenOCRE2E类，创建一个端到端识别器`text_sys = OpenOCRE2E()`并使用`text_sys(img)`对图片进行检测,端到端识别器以list的形式分别返回检测框和对应的检测结果。

### OpenOCR简介

OpenOCR旨在为场景文本检测和识别算法建立统一的训练和评估基准，同时作为复旦大学[FVL](https://fvl.fudan.edu.cn)实验室OCR团队的官方代码库。

我们诚挚欢迎研究人员推荐OCR或相关算法，并指出任何潜在的事实错误或漏洞。在收到建议后，我们会迅速评估并认真验证。我们期待与您合作，共同推进OpenOCR的发展，并持续为OCR社区做出贡献！

### 致谢

这段代码库是基于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)构建的。感谢他们的出色工作！
