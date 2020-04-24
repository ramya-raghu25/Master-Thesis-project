'''
This code is based on https://github.com/zijundeng/pytorch-semantic-segmentation which is licensed under the MIT License.
Copyright (c) 2017 ZijunDeng

# Download links used are also listed here: (https://github.com/pytorch/vision/tree/master/torchvision/models)
DESCRIPTION:     Python script for loading pretrained models
Date: 20.10.2019

For details on the license please have a look at MasterThesis/Licenses/MIT_License.txt
'''

import os
'''
vgg16 trained using caffe
visit this (https://github.com/jcjohnson/pytorch-vgg) to download the converted vgg16
'''
vgg16_caffe_path = os.path.join(os.environ.get('TORCH_MODEL_ZOO', '.'), 'vgg16-caffe.pth')
