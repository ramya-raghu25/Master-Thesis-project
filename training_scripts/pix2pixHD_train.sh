"""
DESCRIPTION:     Shell script for training pix2pixHD GAN
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""


# Change required configs from:
# 	data/base_dataset.py
#	options/base_options.py
#	--fineSize 		= size of square crop
#	--resize_or_crop	


Name='pix2pixHD512_model'

python src/reconstruction/pix2pixHD/train.py --name $Name \
	--checkpoints_dir "Z:/data/samples_osr_yolo/GAN/exp" --dataroot "Z:/Master_Thesis_Ramya/datasets/cityscapes_small/images" \
	--no_instance \
	--resize_or_crop crop --fineSize 384 --batchSize 4 \
	--tf_log
