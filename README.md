# Real3D-Aug



<img align="right" src="images/image1.jpeg" width=40%>

`Real3D-Aug` is a open source project for 3D object detection and semantic segmentation.  

Official paper is published on [arxiv](https://arxiv.org/abs/2206.07634).

`Real3D-Aug` was proposed as lidar augmentation framework, which reuses real data and automatically finds suitable
placements in the scene to be augmented, and handles occlusions explicitly. Due to the usage of the real data,
the scan points of newly inserted objects in augmentation sustain the physical characteristics of the lidar,
such as intensity and raydrop.



<br clear="right"/>

### Overview of the repo

- [Introduction](Introduction)

- [Content](##Content)

- [Demo](##Demo)

- [Licence](##Licence)

- [Acknowledgement](##Acknowledgement)

- [Contribution](##Contribution)

## Introduction

Object detection and semantic segmentation with
the 3D lidar point cloud data require expensive annotation. We
propose a data augmentation method that takes advantage of
already annotated data multiple times. We propose an augmenta-
tion framework `Real3D-Aug`. 


### Proposed pipeline 

The pipeline proves competitive in training top-performing models
for 3D object detection and semantic segmentation. The new
augmentation provides a significant performance gain in rare
and essential classes, notably 6.65% average precision gain for
“Hard” pedestrian class in KITTI object detection or 2.14 mean
IoU gain in the SemanticKITTI segmentation challenge over the
state of the art.


As it is shown on image below, the process of augmentation is divided into 4 steps.

1. **Preprocessing** - In the first step we need to create Rich map if it is not provided. For semantic segmentation dataset we also provide method how to create bounding boxes, which are necessary in further stages.
2. **Placing** In this stage the possible placements are found.
3. **Occlusion handling in spherical coordinates** - To ensure the reality of scanning the occlusion is handled. 
4. **Output** 

![](images/image2.jpeg)



## Content

```
3D-object-detection
  TODO
  
semantic-segmentation/
  config/
  cut_object/
  Real3D-Aug/
  rich_map/
  
```
## Demo

Can run augmentation the procedure by running the python script TASK/Real3D-Aug/insertion.py with modified path to the original dataset.


## Licence

## Acknowledgement

## Citation

## Contribution