"""
DESCRIPTION:     Python script for pre-processing cityscapes dataset
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from ..paths import cityscapes_dir, cityscapes_compressed_dir
import numpy as np
import os
from .dataset import DatasetBase, imread, ChannelLoaderImg, DatasetLabelInfo

# Labels as defined by the dataset
from .cityscapes_labels import labels as cityscapes_labels
CityscapesLabelInfo = DatasetLabelInfo(cityscapes_labels)

# Loader for cityscapes dataset
class DatasetCityscapes(DatasetBase):
	name = 'cityscapes'
	label_info = CityscapesLabelInfo

	def __init__(self, dir_root=cityscapes_dir, split='train', img_ext='.webp', b_cache=True):
		"""
		:param split: Available splits: "train", "test", "val"
		"""
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		self.add_channels(
			image = ChannelLoaderImg(
				img_ext = img_ext,
				temp_path='{dset.dir_root}/images/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/images/gtFine/{dset.split}/{fid}_gtFine_labelIds{channel.img_ext}',
			),
			instances = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/images/gtFine/{dset.split}/{fid}_gtFine_instanceIds{channel.img_ext}',
			),
		)

		self.channel_disable('instances')

		self.post_load_pre_cache.append(
			self.label_info.labelSource_to_trainId,
		)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(
			self.dir_root / 'images' / 'leftImg8bit' / self.split,
			suffix = '_leftImg8bit' + self.channels['image'].img_ext,
		)
		self.load_roi()

		super().discover()

	def load_roi(self):
		# Load a ROI which excludes the ego-vehicle
		roi_path = os.path.join(cityscapes_dir, 'roi.png')
		self.roi = imread(roi_path) > 0
		self.roi_frame = dict(
			roi=self.roi
		)

	@staticmethod
	def get_unknown_gt(labels_source, **_):
		# Load the label map
		return dict(
			unknown_gt = labels_source >= 2,
			roi_onroad=labels_source == 1,
		)

	def get_roi_frame(self, **_):
		return self.roi_frame

	@staticmethod
	def calc_dir_img(dset):
		return dset.dir_root / 'leftImg8bit' / dset.split

	@staticmethod
	def calc_dir_label(dset):
		return dset.dir_root / 'gtCoarse' / dset.split

# Loader for cityscapes compressed dataset
class DatasetCityscapesCompressed(DatasetCityscapes):
	def __init__(self, dir_root=cityscapes_compressed_dir, split='train', b_cache=True):
		super().__init__(dir_root=dir_root, split=split, b_cache=b_cache)

	def load_roi(self):
		#Load a ROI which excludes the ego-vehicle
		roi_path = os.path.join(cityscapes_dir, 'roi.png')
		self.roi = imread(roi_path).astype(np.bool)
		self.roi_frame = dict(
			roi=self.roi
		)