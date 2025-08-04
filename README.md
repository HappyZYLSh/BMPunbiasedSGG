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
python train.py -mode predcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 -lr 1e-5 -save_path output/ 

```

+ For SGCLS:

```
python train.py -mode sgcls -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.3  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```

+ For SGDET:

```
python train.py -mode sgdet -datasize large -data_path $DATAPATH -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -obj_con_loss euc_con  -lambda_con 1  -eos_coef 1 -K 4 -tracking -lr 1e-5 -save_path output/ 

```

## Evaluation

[Trained Models](https://drive.google.com/drive/folders/1m1xSUbqBELpogHRl_4J3ED7tlyp3ebv8?usp=share_link)

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

## Citation

If our work is helpful for your research, please cite our publication:

```
@inproceedings{nag2023unbiased,
  title={Unbiased Scene Graph Generation in Videos},
  author={Nag, Sayak and Min, Kyle and Tripathi, Subarna and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22803--22813},
  year={2023}
}
```
