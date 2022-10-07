# ActiveStereoNet
This repository builds upon the open source pytorch [Active stereo net implementation](https://github.com/linjc16/ActiveStereoNet) and extends it to active tartanair and D435i datsets.

#### Paper
[ActiveStereoNet: End-to-End Self-Supervised Learning for Active Stereo Systems](https://arxiv.org/pdf/1807.06009.pdf)

## Requirments

###### CUDA = v11.1
###### CuDNN >= v8.2.1
###### Python > 3.8
###### Pytorch
###### Torchvision

## Dataset

Datasets used:
Active TartanAir dataset (virtual data)
IMX686 (active real data)

Please, use the links provided to download the datasets and update the ```data_root``` field in the ```Options/*.json``` files.
My take off slamcore version
