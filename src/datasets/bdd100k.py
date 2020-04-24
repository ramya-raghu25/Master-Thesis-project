"""
DESCRIPTION:     Python script for pre-processing BDD100 dataset
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from .dataset import *
from ..paths import BDD_dir
from pjval_ml import PJVAL_SHARE

# Labels as defined by the dataset
from .bdd100k_labels import labels as bdd_labels
BDDLabelInfo = DatasetLabelInfo(bdd_labels)

# Loader for bdd100k dataset
class DatasetBDD_Segmentation(DatasetBase):
	name = 'bdd100k'

	def __init__(self, dir_root=BDD_dir, split='train', b_cache=True, b_recreate_original_ids=False):
		"""
		:param split: Available splits: "train", "test", "val"
		"""
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		self.label_info = BDDLabelInfo
		self.dir_out = Path(PJVAL_SHARE, 'data','samples_osr_yolo','GAN','exp','training_example')
		self.add_channels(
			image = ChannelLoaderImg(
				img_ext = '.jpg',
				temp_path='{dset.dir_root}/images/{dset.split}/{fid}{channel.img_ext}',
			),
			# The label values are the train ids, because that is the only provided format
			labels_source = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/labels/{dset.split}/{fid}_train_id{channel.img_ext}',
			),
		)


	def discover(self):
		self.frames = self.discover_directory_by_suffix(
			self.dir_root / 'images' / self.split,
			suffix = self.channels['image'].img_ext,
		)
		super().discover()

	@staticmethod
	def get_unknown_gt(labels_source, **_):
		return dict(
			unknown_gt = labels_source >= 2,
		)
	@staticmethod
	def get_roi_frame(**_):
		return dict(roi=None)