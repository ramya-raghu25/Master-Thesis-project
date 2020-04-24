"""
DESCRIPTION:     Python script for pre-processing NYU dataset
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from .dataset import *
from ..paths import cityscapes_dir,NYU_dir

# Labels as defined by the dataset in https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
class NYUD_LabelInfo_Category40:
	names = ["unlabeled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop"]

	colors = np.array([
		[0, 0, 0],
		[165,66,104],
		[83,194,76],
		[172,80,199],
		[142,188,58],
		[120,101,223],
		[65,146,51],
		[207,67,168],
		[88,197,124],
		[219,62,131],
		[88,200,169],
		[208,63,54],
		[59,188,195],
		[222,112,55],
		[91,123,222],
		[188,176,57],
		[121,84,172],
		[110,141,48],
		[218,127,219],
		[53,124,65],
		[157,79,149],
		[219,151,53],
		[85,99,164],
		[106,111,27],
		[179,146,214],
		[107,146,81],
		[206,70,94],
		[94,169,121],
		[221,131,174],
		[35,100,63],
		[222,127,120],
		[89,191,236],
		[162,84,46],
		[112,151,218],
		[142,111,44],
		[59,144,191],
		[213,161,107],
		[46,138,112],
		[139,74,95],
		[163,181,109],
		[98,106,50],
	], dtype=np.uint8)

	name2id = { name: idx for idx, name in enumerate(names) }

# Loader for NYU Depth-v2 dataset
class DatasetNYUDv2(DatasetBase):
	name = 'NYUv2'

	def __init__(self, dir_root=NYU_dir, split=None, b_cache=True):
		"""
		:param split: Available splits: "None"
		"""
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split

		self.add_channels(
			image = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/images/nyu_rgb_{fid}{channel.img_ext}',

			),
			labels_source = ChannelLoaderImg(
				img_ext = '.png',
				temp_path='{dset.dir_root}/labels/new_nyu_class13_{fid}{channel.img_ext}',
			),
		)

		self.channel_disable('instances')

	#not required
	def load_roi(self):
		#Load a ROI which excludes the ego-vehicle
		roi_path = os.path.join(cityscapes_dir, 'roi.png')
		self.roi = imread(roi_path) > 0

		self.roi_frame = dict(
			roi=self.roi
		)
	@staticmethod
	def get_unknown_gt(labels_source, **_):
		return dict(
			unknown_gt = labels_source >= 2,
			roi_onroad=labels_source == 1,
		)

	def get_roi_frame(self, **_):
		return self.roi_frame

	def discover(self):

		fids = range(1, 121)

		self.frames = [Frame(fid="%04d" % fid) for fid in fids]
		self.load_roi()
		super().discover()

