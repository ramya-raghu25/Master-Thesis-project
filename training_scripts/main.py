"""
DESCRIPTION:     Main Python file for Unknown Object Segmentation Network
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""


from src.datasets.lost_and_found import DatasetLostAndFound, DatasetLostAndFoundCompressed
from src.datasets.cityscapes import DatasetCityscapes,DatasetCityscapesCompressed
#from src.datasets.bdd100k import DatasetBDD_Segmentation
from src.datasets.NYU_depth_v2 import DatasetNYUDv2
from src.datasets.coco import DatasetCOCO
from src.uosn.A_main_evaluation import UnknownObjectSegmentation
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':

    # can load train or test set
    dset = DatasetLostAndFoundCompressed(split ='test')
    #dset  = DatasetCityscapesCompressed(split='val')
    #dset  = DatasetBDD_Segmentation(split='val')
    #dset = DatasetNYUDv2(split=None)
    #dset = DatasetCOCO(split=None)

    dset.discover()

    #load semseg network :  #BaySegnet  or #PSPnet
    #change line 230,234,248,263,264 in A_main_evaluation
    eval = UnknownObjectSegmentation(semseg_variants = 'PSPnet')
    eval.storage

    #initiliaze semseg architecture
    eval.initialize_semseg()

    # run semseg architecture
    eval.eval_semseg(dset)

    #initiliaze pix2pixHD architecture
    eval.initialize_gan()

    #run pix2pixHD architecture
    eval.eval_gan(dset)

    # run all detectors and store scores
    eval.eval_uosn_all(dset)

    #create directories and image grids to store images
    eval.store_images(dset)

    #eval_obj.anomaly_detector_variants
    eval.unknown_variants

    #calculate ROC
    rocinfos = eval.calc_roc_curves_for_variants(dset)

    #plot ROC curve
    eval.plot_roc(dset)

    print("The end!!")