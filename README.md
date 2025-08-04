# Balancing Minority Prototypes for Long-Tailed Scene Graph Generation

Official Pytorch Implementation of the framework **BMPSGG** proposed in our paper [**Balancing Minority Prototypes for Long-Tailed Scene Graph Generation**]() .

## Overview

The inherent challenges in dynamic scene graph generation, such as long-tailed distribution of the visual relationships, noisy annotations and temporal fluctuation of model predictions, makes existing methods prone to generate biased scene graphs.This module introduces minority focal loss, which enhances their impacts by considering positive/negative factors and prototype probability. To find scarce hard positive samples, we use a progressive way. Then, these progressive samples balance the minority prototypes step-by-
step to ensure their class separability. Extensive experiments demonstrate that our BMP achieves state-of-the-art performance on two scene graph generation datasets, including the Action Genome and the Visual Genome..

![GitHub Logo](/data/framework.png)

## Requirements

Please install packages in the ``environment.yml`` file.

## Usage

We borrow some compiled code for bbox operations.

```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```

For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in

```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset

We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:

```
|-- ag
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```

 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ``dataloader``

## Train

+ For PREDCLS:

```
python train.py -mode predcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 -lr 1e-5   -nstage 6 -alpha 0.05 -positive_gamma 0.5 -negtive_gamma 2 -lambda_consist 100 -lambda_mf 2 -lambda_base 1 -save_path output/ 

```

+ For SGCLS:

```
python train.py -mode sgcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5   -nstage 6 -alpha 0.05 -positive_gamma 0.5 -negtive_gamma 2 -lambda_consist 100 -lambda_mf 2 -lambda_base 1 -save_path output/ 

```

+ For SGDET:

```
python train.py -mode sgdet -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5  -nstage 6 -alpha 0.05 -positive_gamma 0.5 -negtive_gamma 2 -lambda_consist 100 -lambda_mf 2 -lambda_base 1 -save_path output/ 

```

## Evaluation

[Trained Models]()

+ For PREDCLS:

```
python test.py -mode predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 

```

+ For SGCLS:

```
python test.py -mode sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -K 4 -tracking 

```

+ For SGDET:

```
python test.py -mode sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 4 -tracking 

```

## Acknowledgments

We would like to acknowledge the authors of the following repositories from where we borrowed some code

+ [Yang&#39;s repository](https://github.com/jwyang/faster-rcnn.pytorch)
+ [Zellers&#39; repository](https://github.com/rowanz/neural-motifs)
+ [Cong&#39;s repository](https://github.com/yrcong/STTran.git)
+ [Sayak&#39;s repository](https://github.com/sayaknag/unbiasedSGG.git)

## Citation

If our work is helpful for your research, please cite our publication:

```
cff-version: 1.2.0
title: BMP
message: >-
  If you use this software, please cite it using the
  metadata from this file.
type: software
authors:
  - given-names: Shenghao
    family-names: Li
    email: 2023170761@mail.hfut.edu.cn
repository-code: 'https://github.com/HappyZYLSh/BMPunbiasedSGG'
abstract: >-
  The inherent challenges in dynamic scene graph generation,
  such as long-tailed distribution of the visual
  relationships, noisy annotations and temporal fluctuation
  of model predictions, makes existing methods prone to
  generate biased scene graphs.This module introduces
  minority focal loss, which enhances their impacts by
  considering positive/negative factors and prototype
  probability. To find scarce hard positive samples, we use
  a progressive way. Then, these progressive samples balance
  the minority prototypes step-by- step to ensure their
  class separability. Extensive experiments demonstrate that
  our BMP achieves state-of-the-art performance on two scene
  graph generation datasets, including the Action Genome and
  the Visual Genome..
keywords:
  - UnbiasedSGG
license: AAL
commit: XXXXXXXX
version: '1.0'
date-released: '2025-08-04'

```
