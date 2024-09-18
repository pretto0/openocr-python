# OpenOCR

OpenOCR aims to establish a unified training and evaluation benchmark for scene text detection and recognition algorithms, at the same time, serves as the official code repository for the OCR team from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University.

We are actively developing and refining it and expect to release the first version as soon as possible.

We sincerely welcome the researcher to recommend OCR or relevant algorithms and point out any potential factual errors or bugs. Upon receiving the suggestions, we will promptly evaluate and critically reproduce them. We look forward to collaborating with you to advance the development of OpenOCR and continuously contribute to the OCR community!

### Contributors

______________________________________________________________________

Yiming Lei ([pretto0](https://github.com/pretto0)) and Xingsong Ye ([YesianRohn](https://github.com/YesianRohn)) from the [FVL](https://fvl.fudan.edu.cn) Laboratory, Fudan University, under the guidance of Professor Zhineng Chen, completed the majority of the algorithm reproduction work. Grateful for their outstanding contributions.

______________________________________________________________________

### 训练准备

- 修改数据集路径

```
Train:
  dataset:
    name: LMDBDataSet
    data_dir: Path to train data
...

Eval:
  dataset:
    name: LMDBDataSet
    data_dir: Path to eval data
```

- 启动训练

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrnet_ctc.yml
```

### 添加新算法

参考[提交PR流程](https://github.com/Topdu/OpenOCR/pull/2)

流程为：

1、先Fork OpenOCR 项目到自己Github仓库中。

2、git clone -b develop https://github.com/自己的用户名/OpenOCR.git （注意每次git clone 之前要保证自己的仓库是最新代码）。

3、参考svtrnet_ctc和svtr_base_cppd的代码结构，将新算法的preprocess、modeling.encoder、modeling.decoder、optimizer、loss、postprocess添加到代码中。

4、安装pre-commit，执行代码风格检查。

```
pip install pre-commit
pre-commit install
```

5、将新添加的算法训练、评估、测试跑通后，按照github提交commit的流程向源仓库提交PR。

## STD

## E2E

# Acknowledgement

This codebase is built based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR), and [MMOCR](https://github.com/open-mmlab/mmocr). Thanks for their awesome work!
