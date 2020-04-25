"""
DESCRIPTION:     Python file specifying paths for all datasets
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

import os
from pathlib import Path
from pjval_ml import PJVAL_SHARE

exp_dir = os.path.join(PJVAL_SHARE, 'data', 'samples_osr_yolo', 'GAN', 'exp')
#data_dir = os.path.join(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets')
data_dir = os.path.join(PJVAL_SHARE, 'data','samples_osr_yolo','GAN','dataset')

LAF_dir = Path(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'LAF')
LAF_compressed_dir = Path(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'LAF_compressed')

cityscapes_dir = Path(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'cityscapes')
cityscapes_compressed_dir = Path(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'cityscapes_compressed')

#BDD_dir = Path(PJVAL_SHARE, 'data', 'samples_osr_yolo', 'GAN', 'dataset', 'bdd100k')

NYU_dir = Path(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'NYU')

COCO_dir = os.path.join(PJVAL_SHARE, 'data', 'samples_osr_yolo', 'GAN', 'dataset', 'COCO')
