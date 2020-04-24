"""
DESCRIPTION:     Python script for pre-processing lost and found dataset
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

The dataset has been downloaded from http://www.6d-vision.com/lostandfounddataset
For details on the license please have a look at MasterThesis/Licenses/Dataset_License.txt

"""


import logging
import numpy as np
import os, re
from .dataset import DatasetBase, imread,ChannelLoaderImg
from ..paths import LAF_dir,LAF_compressed_dir,cityscapes_dir
log = logging.getLogger('exp')


# Loader for LostAndFound dataset
class DatasetLostAndFound(DatasetBase):

	name = 'lost_and_found'
	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']

	# invalid frames are those where np.count_nonzero(labels_source) is 0
	INVALID_LABELED_FRAMES = {
		'train': [44, 67, 88, 109, 131, 614],
		'test': [17, 37, 55, 72, 91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
	}
	def __init__(self, dir_root=LAF_dir, split='train', only_interesting=True, only_valid=True, b_cache=True):
		"""
		:param split: Available splits: "train", "test"
		:param only_interesting: takes the last frame from each sequence, i.e the frame in which the object is the closest to the camera
		:param only_valid: excludes the INVALID_LABELED_FRAMES
		"""
		super().__init__(b_cache=b_cache)
		self.dir_root = dir_root
		self.split = split
		self.only_interesting = only_interesting
		self.only_valid = only_valid

		self.add_channels(
			image = ChannelLoaderImg(
				temp_path='{dset.dir_root}/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_labelIds{channel.img_ext}',
			),
			instances = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_instanceIds{channel.img_ext}',
			),
		)

	def load_roi(self):
		#Load a ROI which excludes the ego-vehicle
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

	def discover(self):
		for img_ext in self.IMG_FORMAT_TO_CHECK:
			self.frames_all = self.discover_directory_by_suffix(
				self.dir_root / 'leftImg8bit' / self.split,
				suffix=f'_leftImg8bit{img_ext}',
			)

			if self.frames_all:
				log.info(f'LAF: found images in {img_ext} format')
				break

		self.channels['image'].img_ext = img_ext
		# LAF's images contain a gamma value which makes them too bright, so ignore it
		if img_ext == '.png':
			self.channels['image'].opts['ignoregamma'] = True

		# parse names to determine scenes, sequences and timestamps
		for fr in self.frames_all:
			fr.apply(self.laf_name_to_sc_seq_t)

		#excludes frames from the INVALID_LABELED_FRAMES
		if self.only_valid:
			invalid_indices = self.INVALID_LABELED_FRAMES[self.split]
			valid_indices = np.delete(np.arange(self.frames_all.__len__()), invalid_indices)
			self.frames_all = [self.frames_all[i] for i in valid_indices]

		# organize frames into hierarchy
		scenes_by_id = dict()

		for fr in self.frames_all:
			scene_seqs = scenes_by_id.setdefault(fr.scene_id, dict())
			seq_times = scene_seqs.setdefault(fr.scene_seq, dict())
			seq_times[fr.scene_time] = fr

		self.frames_interesting = []

		# Select the last frame in each sequence, when the object is the closest to the camera
		for sc_name, sc_sequences in scenes_by_id.items():
			for seq_name, seq_times in sc_sequences.items():
				t_last = max(seq_times.keys())
				self.frames_interesting.append(seq_times[t_last])

		# takes the last frame from each sequence
		self.use_only_interesting(self.only_interesting)
		self.load_roi()
		super().discover()


	RE_LAF_NAME = re.compile(r'([0-9]{2})_.*_([0-9]{6})_([0-9]{6})')
	"""
	This extracts the scene, sequence, and time-within-sequence from a LostAndFound name.
	Given a name 01_Hanns_Klemm_Str_45_000000_000200_leftImg8bit, these are:
		scene: 01
		sequence: 000000
		time: 000200
	"""

	@staticmethod
	def laf_name_to_sc_seq_t(fid, **_):
		m = DatasetLostAndFound.RE_LAF_NAME.match(fid)

		return dict(
			scene_id = int(m.group(1)),
			scene_seq = int(m.group(2)),
			scene_time = int(m.group(3))
		)

	def use_only_interesting(self, only_interesting):
		self.only_interesting = only_interesting
		self.frames = self.frames_interesting if only_interesting else self.frames_all

# Loader for LostAndFound compressed dataset
class DatasetLostAndFoundCompressed(DatasetLostAndFound):
	def __init__(self, dir_root=LAF_compressed_dir, **kwargs):
		super().__init__(dir_root=dir_root, **kwargs)

	def load_roi(self):
		# Load a ROI which excludes the ego-vehicle and registration artifacts
		roi_path = os.path.join(cityscapes_dir, 'roi.png')
		self.roi = imread(roi_path).astype(np.bool)
		self.roi_frame = dict(
			roi=self.roi
		)
