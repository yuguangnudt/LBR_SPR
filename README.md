## Deep Anomaly Discovery from Unlabeled Videos via Normality Advantage and Self-Paced Refinement

Official implementation of ["Deep Anomaly Discovery from Unlabeled Videos via Normality Advantage and Self-Paced Refinement"](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.html) accepted by CVPR 2022.

### 1. Dependencies

```
ubuntu 16.04
cuda 10.1
cudnn 7.6.4
python 3.7
pytorch 1.7.0
torchvision 0.8.0
numpy 1.19.2
opencv-contrib-python 4.1.1.26
mmdetection 2.11.0
mmcv-full 1.3.1
```

### 2. Preparing Data

(1) Download and organize VAD datasets: Download UCSDped1/ped2 from [official source](http://svcl.ucsd.edu/projects/anomaly/dataset.htm) and complete pixel-wise ground truth of UCSDped1 from the [website](https://hci.iwr.uni-heidelberg.de/content/video-parsing-abnormality-detection); Avenue and ShanghaiTech from [OneDrive](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F) or [BaiduNetdisk](https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ) (code: i9b3, provided by [StevenLiuWen](https://github.com/StevenLiuWen/ano_pred_cvpr2018)) , and [ground truth](www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip) of avenue from [official source](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html). Then create a folder named `raw_datasets` in root directory to place the downloaded datasets. The directory structure should be organized according to `tree.txt`.

(2) Calculate optical flow: Follow the [instructions](https://github.com/vt-vl-lab/flownet2.pytorch) to compile FlowNet2. Then download and move the pretrained [FlowNet 2.0](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing) (`FlowNet2_checkpoint.pth.tar`) to `./FlowNet2_src/pretrained`.  Finally run (in PyTorch 0.3.0): `python calc_optical_flow.py --dataset_name {UCSDped1, UCSDped2, avenue or ShanghaiTech} --mode {train or test}`. As an alternative, you can follow this [repository](https://github.com/LiUzHiAn/hf2vad/tree/master/pre_process) to extract optical flow to avoid the issues related to PyTorch being too old.

(3) Localize foreground objects: Follow the [instructions](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) to install mmdetection (v2.11.0). Then download the pretrained object detector [YOLOv3](download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth) in this [page](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/configs/yolo/README.md), and move it to `fore_det/obj_det_checkpoints` . Finally run: `python extract_bbox.py --dataset_name {UCSDped1, UCSDped2, avenue or ShanghaiTech} --mode {test or merge}`.

(4) Extract STCs for training and testing: `python extract_stc.py --{extract_training_stc or extract_testing_stc} --dataset_name {UCSDped1, UCSDped2, avenue or ShanghaiTech} --mode {test or merge}` .

### 3. Training

```python
python train.py --dataset_name {UCSDped1, UCSDped2, avenue or ShanghaiTech} --mode {test or merge}
```

`mode`: The testing set or the merged set of training and testing set of a VAD dataset. Here the training set that only contains normal videos should not be used alone due to the unsupervised setting (i.e. UVAD).

### 4. Testing

```python
python test.py --dataset_name {UCSDped1, UCSDped2, avenue or ShanghaiTech} --mode {test or merge}
```

To facilitate testing and verification, we provide the models (without motion enhancement) trained on testing set (i.e. the partial mode, see Sec. 4.1 of our paper). Since the provided models are trained under partial mode, please set mode=test to test them.

### 5. Citation

```
@InProceedings{Yu_2022_CVPR,
    author    = {Yu, Guang and Wang, Siqi and Cai, Zhiping and Liu, Xinwang and Xu, Chuanfu and Wu, Chengkun},
    title     = {Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {13987-13998}
}
```
