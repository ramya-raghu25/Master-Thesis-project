"""
DESCRIPTION:     Python script for helper functions(loading, saving etc.) for all datasets
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

import logging
log = logging.getLogger('exp')
from pathlib import Path
import numpy as np
import h5py, threading, os
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset as PytorchDatasetInterface
from ..pipeline.utils import hdf5_load, hdf5_save
from ..pipeline.frame import Frame
from ..pipeline.transforms import Chain, ByField

def imread(path):
	return np.asarray(Image.open(path))

WRITE_OPTS = dict(
	webp = dict(quality = 90),
)

# image saving is asynchronous. Do the following to see the exception
def imwrite(path, data):

	try:
		path = Path(path)
		Image.fromarray(data).save(
			path,
			**WRITE_OPTS.get(path.suffix.lower()[1:], {}),
		)
	except Exception:
		log.exception(f'Error while saving image {path}')

#write hdf5 files
def hdf5_write_recursive(handle, subpath, data):
	if isinstance(data, dict):
		g = handle.require_group(subpath)
		for name, value in data.items():
			hdf5_write_recursive(g, name, value)
	elif isinstance(data, str):
		handle.attrs[subpath] = data
	else:
		handle[subpath] = data

#write hdf5 files
def hdf5_write(path, data):
	with h5py.File(path, 'w') as file:
		for name, value in data.items():
			hdf5_write_recursive(file, name, value)

#read hdf5 files
def hdf5_read_recursive(group):

	value_dict = dict(group.attrs) 	# for string data

	for k, obj in group.items():  	# for subgroups and numerics

		if isinstance(obj, h5py.Group):
			val = hdf5_read_recursive(obj)
		elif isinstance(obj, h5py.Dataset):
			val = obj[()]

		value_dict[k] = val

	return value_dict

#read hdf5 files
def hdf5_read(path):
	with h5py.File(path, 'r') as file:
		return hdf5_read_recursive(file)

LOCK_HDF_CREATION = threading.Lock()

class DatasetBase(PytorchDatasetInterface):

	def __init__(self,  b_cache=True):
		self.b_cache = b_cache
		self.frames = []
		self.frame_idx_by_fid = {}
		self.channels = dict()
		self.channels_enabled = set()

		# Call the required Transform funtions
		self.post_load_pre_cache = Chain()
		self.tr_output = Chain()
		self.hdf5_files = dict()
		self.fake_num_frames = None

	#Count the number of frames
	def after_discovering_frames(self):
		self.frame_idx_by_fid = {fr.fid: idx for (idx, fr) in enumerate(self.frames)}
		for fr in self.frames:
			fr['dset'] = self

	def discover(self):
		""" Discovers all frames and stores them in self.frames.
			Each frame has the frame id "fid" field to uniquely identify it within the specific dataset
		"""
		self.after_discovering_frames()
		log.info(f'Discovered {self.__len__()} frames from {self}')
		return self

	def discover_directory_by_suffix(self, src_dir, suffix=''):
		suffix_len = suffix.__len__()

		return [
			Frame(fid = fn[:-suffix_len]) # removes suffix to get only the ID
			for fn in listdir_recursive(src_dir, ext=suffix, relative_paths=True)
		]

	def add_channels(self, **chs):
		for name, loader in chs.items():
			self.channels[name] = loader
			self.channels_enabled.add(name)

	def check_channel(self, ch):
		if ch not in self.channels:
			raise KeyError('Requested to enable channel {ch} but the dataset does not have it (channels = {chs})'.format(
				ch=ch,
				chs = ', '.join(self.channels.keys())
			))

	def set_enabled_channels(self, *chns):
		"""
		e.g: dset.set_enabled_channels(["image", "labels"]) enables only images and labels
		(or)
		set_enabled_channels('*') enables all
		"""
		if '*' in chns:
			chns = list(self.channels.keys())
		elif chns.__len__() == 1 and (isinstance(chns[0], list) or isinstance(chns[0], tuple)):
			chns = chns[0]

		self.channels_enabled = set()
		self.channel_enable(*chns)

	#enable the specified channels
	def channel_enable(self, *chns):
		for ch in chns:
			self.check_channel(ch)
			self.channels_enabled.add(ch)

	#disable the specified channels
	def channel_disable(self, *chns):
		self.channels_enabled.difference_update(chns)

	def load_frame(self, frame):
		for ch_name in self.channels_enabled:
			if ch_name not in frame:
				self.channels[ch_name].load(self, frame, ch_name)

		frame.apply(self.post_load_pre_cache)

	def save_frame_channel(self, frame, ch):
		self.channels[ch].save(self, frame, ch)

	def __getitem__(self, index):
		if isinstance(index, str):
			fr = self.get_frame_by_fid(index)
		else:
			fr = self.frames[index]

		if self.b_cache:
			if not fr.get('cached', False):
				self.load_frame(fr)
				fr.cached = True
			out_fr = fr.copy()
			del out_fr['cached']
		else:
			out_fr = fr.copy()
			self.load_frame(out_fr)

		out_fr.apply(self.tr_output)

		return out_fr

	def get_frame_by_fid(self, fid):
		return self[self.frame_idx_by_fid[fid]]

	def get_idx_by_fid(self, fid):
		return self.frame_idx_by_fid[fid]

	def __len__(self):
		return self.fake_num_frames or self.frames.__len__()

	def __iter__(self):
		for idx in range(self.__len__()):
			yield self[idx]

	def __repr__(self):
		dr = getattr(self, 'dir_root', None)
		split = getattr(self, 'split', None)
		part = getattr(self, 'part', None)

		return '{cn}({nf} frames{rd}{split}{part})'.format(
			cn = self.__class__.__name__,
			nf = self.__len__(),
			rd = ', ' + str(dr) if dr is not None else '',
			split = ', s=' + split if split is not None else '',
			part = ', part=' + part if part is not None else '',
		)


	def get_hdf5_file(self, file_path, write=False):
		file_path = Path(file_path)

		with LOCK_HDF_CREATION:
			handle = self.hdf5_files.get(file_path, None)

			if handle is None:
				if write:
					file_path.parent.mkdir(exist_ok=True)

				mode = (
					('a' if file_path.is_file() else 'w') if write
					else 'r'
				)

				handle = h5py.File(file_path, mode)
				self.hdf5_files[file_path] = handle

			return handle

	def out_hdf5_files(self):
		for handle in self.hdf5_files.values():
			handle.close()
		self.hdf5_files = dict()

	def original_dataset(self):
		return self


	def load_class_statistics(self, path_override=None):
		path = path_override or '{dset.dir_out}/class_stats/{dset.split}_stats.hdf5'.format(dset=self)
		self.class_statistics = Frame(hdf5_load(path))

	def save_class_statistics(self, path_override=None):
		path = path_override or '{dset.dir_out}/class_stats/{dset.split}_stats.hdf5'.format(dset=self)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		hdf5_save(path, self.class_statistics)

class ChannelLoader:
	def load(self, dset, frame, field_name):
		raise NotImplementedError()

# Loader for each channel
class ChannelLoaderFileCollection:
	"""
	The channel's value for each frame is in a separate file, for example an image.
	:param temp_path: a template string or a function(channel, dset, fid)
	"""
	def __init__(self, temp_path):
		# to convert Path to string, use .format
		self.temp_path = str(temp_path) if isinstance(temp_path, Path) else temp_path

	def resolve_template(self, template, dset, frame):
		if isinstance(template, str):	# string template
			#print("string template")
			return template.format(dset=dset, channel=self, frame=frame, fid=frame['fid'], remove_slash=frame['fid'].replace('/', '__'))
		else:			               # function template
			#print("function template")
			return template(dset=dset, channel=self, frame=frame, fid=frame['fid'])

	def resolve_file_path(self, dset, frame):
		return Path(self.resolve_template(self.temp_path, dset, frame))

	def load(self, dset, frame, field_name):
		path = self.resolve_file_path(dset, frame)
		frame[field_name] = self.read_file(path)

	def save(self, dset, frame, field_name):
		path = self.resolve_file_path(dset, frame)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		self.write_file(path, frame[field_name])

	def read_file(self, path):
		raise NotImplementedError('Reading file for {c}'.format(c=self.__class__.__name__))

	def write_file(self, path, data):
		raise NotImplementedError('Writing file for {c}'.format(c=self.__class__.__name__))

	def discover_files(self, dset):
		pattern = self.resolve_file_path(dset, Frame(fid='**'))
		return glob.glob(pattern)

	def __repr__(self):
		return '{cls}({tp})'.format(cls=self.__class__.__name__, tp=self.temp_path)


class ImageBackgroundService:
	IMWRITE_BACKGROUND_THREAD = ThreadPoolExecutor(max_workers=3)

	@classmethod
	def imwrite(cls, path, data):
		cls.IMWRITE_BACKGROUND_THREAD.submit(imwrite, path, data)


class ChannelLoaderImg(ChannelLoaderFileCollection):
	def __init__(self, temp_path=None, dir_root=None, suffix='', img_ext='.jpg'):
		"""
		Specify the root, suffix and image extension here(channel.dir_root or dset.dir_root)/fid + channel.suffix + channel.img_ext
		"""
		super().__init__(temp_path or self.temp_path_generic)

		self.img_ext = img_ext
		self.opts = dict()

		self.dir_root = dir_root
		self.suffix = suffix

	@staticmethod
	def temp_path_generic(channel, dset, fid):
		#returns path based on self.dir_root and self.suffix and img_ext
		return os.path.join(
			channel.dir_root or dset.dir_root,
			fid + channel.suffix + channel.img_ext,
		)

	def read_file(self, path):
		return np.asarray(imread(path, **self.opts))

	def write_file(self, path, data):
		ImageBackgroundService.imwrite(path, data)


class ChannelLoaderHDF5_Base(ChannelLoaderFileCollection):

	def __init__(self, temp_path, var_name_temp='{fid}/{field_name}', index_func=None, compression=None, write_as_type=None, read_as_type = None):
		super().__init__(temp_path)

		self.var_name_temp = var_name_temp
		self.index_func = index_func
		self.compression = compression
		self.write_as_type = write_as_type
		self.read_as_type = read_as_type

	def resolve_var_name(self, dset, frame, field_name):
		return self.var_name_temp.format(dset=dset, channel=self, frame=frame, fid=frame['fid'], field=field_name)

	def resolve_index(self, dset, frame, field_name):
		if self.index_func:
			return self.index_func(dset=dset, channel=self, frame=frame, fid=frame['fid'], field=field_name)
		else:
			return None

	@staticmethod
	def read_hdf5_variable(variable):
		if variable.shape.__len__() > 0:
			return variable[:]
		else:
			return variable

	def load_from_handle(self, dset, frame, field_name, hdf5_file_handle):
		var_name = self.resolve_var_name(dset, frame, field_name)
		if self.index_func:
			index = self.resolve_index(dset, frame, field_name)
			value = self.read_hdf5_variable(hdf5_file_handle[var_name][index])
		else:

			try:
				value = self.read_hdf5_variable(hdf5_file_handle[var_name])
			except KeyError as e:
				raise KeyError(f'Failed to read file')

		if self.read_as_type:
			value = value.astype(self.read_as_type)

		frame[field_name] = value


	def save_to_handle(self, dset, frame, field_name, hdf5_file_handle):

		var_name = self.resolve_var_name(dset, frame, field_name)
		value_to_write = frame[field_name]

		if self.write_as_type:
			value_to_write = value_to_write.astype(self.write_as_type)

		if var_name in hdf5_file_handle:
			if self.index_func:
				index = self.resolve_index(dset, frame, field_name)
				hdf5_file_handle[var_name][index][:] = value_to_write
			else:
				hdf5_file_handle[var_name][:] = value_to_write
		else:
			if self.index_func:
				raise NotImplementedError('Writing to new HDF5 with index_func')

			hdf5_file_handle.create_dataset(var_name, data=value_to_write, compression=self.compression)

#Loading respective HDF5 files
class ChannelLoadHDF5(ChannelLoaderHDF5_Base):

	def load(self, dset, frame, field_name):
		hdf5_file_path = self.resolve_file_path(dset, frame)
		hdf5_file_handle = dset.get_hdf5_file(hdf5_file_path, write=False)
		self.load_from_handle(dset, frame, field_name, hdf5_file_handle=hdf5_file_handle)

	def save(self, dset, frame, field_name):
		hdf5_file_path = self.resolve_file_path(dset, frame)
		hdf5_file_handle = dset.get_hdf5_file(hdf5_file_path, write=True)
		self.save_to_handle(dset, frame, field_name, hdf5_file_handle=hdf5_file_handle)


#Location for saving output images
class ChannelResultImg(ChannelLoaderImg):
	def __init__(self, name, *args, temp_path ='{dset.dir_out}/{channel.name}/{dset.split}/{fid}{channel.suffix}{channel.img_ext}', **kwargs):
		self.name = name

		super().__init__(
			*args,
			temp_path= temp_path,
			**kwargs,
		)

	def __repr__(self):
		return '{cls} "{n}"'.format(cls=self.__class__.__name__, n=self.name)


def listdir_recursive(base_dir, ext=None, relative_paths=False):
	results = []
	for (dir_cur, dirs, files) in os.walk(base_dir):
		if relative_paths:
			dir_cur = os.path.relpath(dir_cur, base_dir)
		if dir_cur == '.':
			dir_cur = ''

		for fn in files:
			if ext is None or fn.endswith(ext):
				results.append(os.path.join(dir_cur, fn))

	results.sort()
	return results


class SaveChannelsAutoDset:
	def __init__(self, channels, ignore_none=False):
		if isinstance(channels, str):
			self.channels = [channels]
		else:
			self.channels = channels

		if isinstance(ignore_none, str):
			raise RuntimeError(f'channel name {ignore_none} in ignore_none')
		
		self.ignore_none = ignore_none

	def __call__(self, frame, dset, **_):
		if self.ignore_none:
			chs_to_save = [ch for ch in self.channels if ch in frame]
		else:
			chs_to_save = self.channels

		for ch in chs_to_save:
			dset.save_frame_channel(frame, ch)

	def __repr__(self):
		return '{clsn}({args})'.format(
			clsn = self.__class__.__name__,
			args = ', '.join(self.channels),
		)

#Covert black and white images to color images
def binary_color_to_rgb():
	masks = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.int32)
	shifts = np.array([16, 8, 0], dtype=np.int32)

	def binary_color_to_rgb(binary_color):
		return (binary_color & masks) >> shifts

	return binary_color_to_rgb

#Covert black and white images to color images
binary_color_to_rgb = binary_color_to_rgb()


def apply_label_translation_table(table, labels_source):
	mx = np.max(labels_source)
	if mx >= table.shape[0] - 1:
		if mx != 255:
			print('Overflow in labels', mx)
		labels_source = labels_source.copy()
		labels_source[labels_source >= table.shape[0]] = 0

	return table[labels_source.reshape(-1)].reshape(labels_source.shape)

#Identifies which class it belongs to based on the segments
class SemSegLabelTranslation(ByField):
	def __init__(self, table, fields=[('labels_source', 'labels')]):
		super().__init__(fields=fields)
		self.table = table

	def forward(self, field_name, value):
		return apply_label_translation_table(self.table, value)

#data manipulation between labels and trainIDS
class DatasetLabelInfo:
	def __init__(self, label_list):

		self.labels = label_list

		# converts name to label
		self.name2label = {label.name: label for label in self.labels}
		self.name2id = {label.name: label.id for label in self.labels}
		self.name2trainId = {label.name: label.trainId for label in self.labels}
		# converts id to label
		self.id2label = {label.id: label for label in self.labels}
		# converts trainId to label
		self.trainId2label = {label.trainId: label for label in reversed(self.labels)}

		self.tabulate_label_to_trainId = self.build_translation_table([
			(label.id, label.trainId) for label in self.labels
		])
		self.tabulate_trainId_to_label = self.invert_translation_table(self.tabulate_label_to_trainId)
		#converts pixels from label maps to trainId
		self.labelSource_to_trainId = SemSegLabelTranslation(self.tabulate_label_to_trainId)
		self.trainId_to_labelSource = SemSegLabelTranslation(self.tabulate_trainId_to_label)

		self.valid_in_eval = self.build_bool_table([
			(label.id, not label.ignoreInEval) for label in self.labels
		], False)
		self.valid_in_eval_trainId = self.build_bool_table([
			(label.trainId, not label.ignoreInEval) for label in self.labels
		], False)

		self.num_labels = self.labels.__len__()
		# length is subtracted by 1 since 255(unlabelled) is in the list
		self.num_trainIds = self.trainId2label.__len__() - 1

		self.build_colors()

	@staticmethod
	#creating a table of colors for the respective RGB color code
	def build_color_table(index_color_pairs):
		table_size = max(index for (index, color) in index_color_pairs) + 1
		table_size = max(table_size, 256)

		color_table = np.zeros(
			(table_size, 3),
			dtype=np.uint8,
		)
		#do not use this color
		color_table[:] = (0, 255, 255)

		for idx, color in reversed(index_color_pairs):
			color_table[idx] = color

		return color_table

	@staticmethod
	def build_bool_table(pairs, default_val):
		table_size = max(src for (src, dest) in pairs) + 1
		table_size = max(table_size, 256)

		table = np.zeros(
			table_size,
			dtype=np.bool,
		)
		table[:] = default_val

		for (src, dest) in pairs:
			table[src] = dest

		return table

	@staticmethod
	def build_translation_table(pairs):
		table_size = max(src for (src, dest) in pairs) + 1
		table_size = max(table_size, 256)

		table = np.zeros(
			table_size,
			dtype=np.uint8,
		)
		table[:] = 255

		for (src, dest) in pairs:
			table[src] = dest

		return table

	@staticmethod
	def invert_translation_table(table):

		table_inv = np.zeros(np.max(table) + 1, dtype=table.dtype)

		for src, dest in reversed(list(enumerate(table))):
			table_inv[dest] = src

		return table_inv

	def build_colors(self):
		labels_replace = []
		for label in self.labels:
			if not isinstance(label.color, tuple):
				color_decoded = binary_color_to_rgb(label.color)
				#not used
				try:
					label.color = color_decoded
				except AttributeError:
					label = label._replace(color=color_decoded)

			labels_replace.append(label)

		self.labels = labels_replace

		self.colors_by_id = self.build_color_table([
			(label.id, label.color) for label in self.labels
		])

		self.colors_by_trainId = self.build_color_table([
			(label.trainId, label.color) for label in self.labels
		])
