<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A2-Net]`](https://arxiv.org/abs/1907.03670), [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192), [`[Voxel R-CNN]`](https://arxiv.org/abs/2012.15712), [`[PV-RCNN++]`](https://arxiv.org/abs/2102.00463) and [`[MPPNet]`](https://arxiv.org/abs/2205.05979). 


## Overview
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Design Pattern](#openpcdet-design-pattern)
- [Model Zoo](#model-zoo)
- [Citation](#citation)


## Installation

### Install `pcdet v0.6`
1. Clone this repository.
```shell
git clone https://github.com/Meda1703/OpenPCDet.git
```

2. Install spconv library.
```shell
# python 3.7.11
conda install pytorch==1.10.0 cudatoolkit=10.2 torchvision -c pytorch -y
```

3. Install spconv library.
```shell
pip install spconv-cu102==2.2.3
```

4. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
5. Install requirements
```shell
pip install -r requirements.txt
```

Please refer to [INSTALL.md](docs/INSTALL.md) for detailed installation of `OpenPCDet`.

## Quick Demo

```
pip install open3d
```
Run the demo with a pretrained model (e.g. Voxel-RCNN):

```shell
python demo.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml \
    --ckpt ../checkpoints/voxel_rcnn_car_84.54.pth \
    --data_path ../data/kitti/training/velodyne/000008.bin
```

Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and 
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started


### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI training set
* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 
cd tools
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
cd tools
python train.py --cfg_file ${CONFIG_FILE}
```

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
cd tools
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
cd tools
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
cd tools
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or
cd tools
sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

## Introduction


### What does `OpenPCDet` toolbox do?

Note that we have upgrated `PCDet` from `v0.1` to `v0.2` with pretty new structures to support various datasets and models.

`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

Based on `OpenPCDet` toolbox, we win the Waymo Open Dataset challenge in [3D Detection](https://waymo.com/open/challenges/3d-detection/), 
[3D Tracking](https://waymo.com/open/challenges/3d-tracking/), [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/) 
three tracks among all LiDAR-only methods, and the Waymo related models will be released to `OpenPCDet` soon.    

We are actively updating this repo currently, and more datasets and models will be supported soon. 
Contributions are also welcomed. 

### `OpenPCDet` design pattern

* Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:
<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* Unified 3D box definition: (x, y, z, dx, dy, dz, heading).

* Flexible and clear model structure to easily support various 3D detection models: 
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* Support various models within one framework as: 
<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>


### Currently Supported Features

- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Support distributed training & testing with multiple GPUs and multiple machines
- [x] Support multiple heads on different scales to detect different classes
- [x] Support stacked version set abstraction to encode various number of points in different scenes
- [x] Support Adaptive Training Sample Selection (ATSS) for target assignment
- [x] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [x] Support GPU version 3D IoU calculation and rotated NMS 


## Model Zoo

### KITTI 3D Object Detection Baselines
The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.


|                                                                                                | Car@R11 | Pedestrian@R11 | Cyclist@R11 | Epoch |
|------------------------------------------------------------------------------------------------|--------:|:--------------:|:-----------:|:-----:|
| [Voxel R-CNN (Baseline)](tools/cfgs/kitti_models/voxel_rcnn/voxel_rcnn_car.yaml)               |   83.06 |       -        |      -      |  76   |
| [Voxel R-CNN (Ranger optimizer)](tools/cfgs/kitti_models/voxel_rcnn/voxel_rcnn_ranger.yaml)    |   82.44 |       -        |      -      |  72   |
| [Voxel R-CNN (Asym Encoder, no neck)](tools/cfgs/kitti_models/voxel_rcnn/voxel_rcnn_asym.yaml) |   82.64 |       -        |      -      |  77   |
| [Voxel R-CNN (DA-sampler)](tools/cfgs/kitti_models/voxel_rcnn/voxel_rcnn_da_gt_sampling.yaml)  |   85.14 |       -        |      -      |  75   |
| [Voxel R-CNN (DA-sampler) with wbf](tools/cfgs/kitti_models/voxel_rcnn/voxel_rcnn_da_wbf.yaml) |   85.64 |       -        |      -      |  75   |



## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## Contribution
Welcome to be a member of the OpenPCDet development team by contributing to this repo, and feel free to contact us for any potential contributions. 


