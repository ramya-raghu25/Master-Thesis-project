"""
DESCRIPTION:     Python script for pre-processing MS COCO dataset
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from .dataset import *

import logging
from .dataset import DatasetBase,ChannelLoaderImg
from ..paths import COCO_dir
log = logging.getLogger('exp')

# Loader for MS COCO dataset
class DatasetCOCO(DatasetBase):

	name = 'voc'
	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']
	def __init__(self, dir_root=COCO_dir, split='None', b_cache=True):
		"""
		:param split: Available splits: "None"
		"""
		super().__init__(b_cache=b_cache)
		self.dir_root = dir_root
		self.split = split

		self.add_channels(
			image = ChannelLoaderImg(
				temp_path='{dset.dir_root}/images/{fid}_scene{channel.img_ext}',
			),
			labels_source = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/labels/{fid}_scene{channel.img_ext}',
			),
			instances=ChannelLoaderImg(
				temp_path='{dset.dir_root}/labels/{fid}_inst{channel.img_ext}',
			),

		)

	@staticmethod
	def get_unknown_gt(labels_source, **_):
		# Load the label map

		return dict(
			unknown_gt = labels_source >= 2,
		)

	#not required
	def get_roi_frame(self, **_):
		return dict(roi=None)

	def discover(self):

		fids = range(0,5)
		self.frames = [Frame(fid="%02d" % fid) for fid in fids]
		super().discover()

