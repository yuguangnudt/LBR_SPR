## Deep Anomaly Discovery from Unlabeled Videos via Normality Advantage and Self-Paced Refinement

This repository is the official implementation of "Deep Anomaly Discovery from Unlabeled Videos via Normality Advantage and Self-Paced Refinement" accepted by CVPR 2022.

## 1. Requirements

(a) The basic running environment is as follows:

```
ubuntu 16.04
cuda 10.1
cudnn 7.6.4
python 3.7
pytorch 1.7.0
torchvision 0.8.0
numpy 1.19.2
opencv-contrib-python 4.1.1.26
```

(b) To localize foreground objects, please follow the [instructions](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) to install mmdetection (v2.11.0). Then download the pretrained object detector [YOLOv3](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/configs/yolo/README.md), and move it to `fore_det/obj_det_checkpoints` .

## 2. Data preparation

Download UCSDped1/ped2 from [official source](http://svcl.ucsd.edu/projects/anomaly/dataset.htm) and complete pixel-wise ground truth of UCSDped1 from [website](https://hci.iwr.uni-heidelberg.de/content/video-parsing-abnormality-detection), Avenue and Shanghaitech from [OneDrive](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F) or [BaiduNetdisk](https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ) (code: i9b3, provided by [StevenLiuWen](https://github.com/StevenLiuWen/ano_pred_cvpr2018)) , and [ground truth](www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip) of avenue from [official source](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html). Then create a folder named `raw_datasets` in root directory to store the downloaded datasets. The directory structure should be organized to match `vad_datasets.py` as follows: 

```
.
├── ...
├── raw_datasets
 │   ├── avenue
 │   │   ├── ground_truth_demo
 │   │   ├── testing
 │   │   └── training
 │   ├── ShanghaiTech
 │   │   ├── Testing
 │   │   ├── training
 │   │   └── training.zip
 │   ├── UCSDped1
 │   │   ├── Test
 │   │   └── Train
 │   ├── UCSDped2
 │   │   ├── Test
 │   │   └── Train
├── calc_optical_flow.py
├── ...
```



## 3.  Test on saved models

(a) Please choose the model you want to test in `./data/raw2flow/saved_models`, and copy all the files in this model folder (e.g., avenue_LBR-SPR_partial_mode) to  `./data/raw2flow/`. 

(b) Then edit the file `config.cfg` according to the model you choose and parameters reported in this paper (e.g., `dataset_name`).  

(c) Finally run  `test.py`: `python test.py`.

## 4. Train

Please run `train.py`: `python train.py`. Before training, you can edit the file `config.cfg` according to your own requirements or implementation details reported in this paper.