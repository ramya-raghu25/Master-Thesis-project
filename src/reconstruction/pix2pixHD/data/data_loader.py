'''

This code is based on https://github.com/NVIDIA/pix2pixHD which is licensed
under the BSD License

DESCRIPTION:Python file for pre-processing dataset for pix2pixHD
Date: 20.10.2019

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
All rights reserved.

For details on the license please have a look at MasterThesis/Licenses/pix2pixHD_License.txt
'''

def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
