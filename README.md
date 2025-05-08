# Future vision transport
[![version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/Wonters/future_vision_transport)
## Table of content
- [Description](#description)
- [Installation](#installation)
- [Train](https://www.futurevisiontransport.com/)


## Description
Future vision transport handle to generetate segmented images of 
driving area to give the process of mask generation automatic.
Image segmented generation for autonomous drive is a major challenge because 
of the complexity of masks, the precision required, and the cost of computing.  
This application handle the long task of a manual segmentation if new images.

## Installation
- pip install

## Train on GPU
DEBUG
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```
```bash
python -m torch.distributed.run --nproc_per_node=2 train.py
```
