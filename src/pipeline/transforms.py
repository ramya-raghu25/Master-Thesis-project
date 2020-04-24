"""
DESCRIPTION:     Python script for important transforms performed during pipeline
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""


import logging, torch, inspect, types
log = logging.getLogger('exp')
import numpy as np
from .frame import *
from ..pipeline.utils import show

try:
	import cv2 as cv
except Exception as e:
	log.warning('OpenCV import failed:', e)

#Base class for all transforms in the experiments
class Base:

	@staticmethod
	def object_name(obj):
		return getattr(obj, '__name__', None) or obj.__class__.__name__
	
	@staticmethod
	def repr_with_args(name, args):
		return '{name}({args})'.format(
			name = name,
			args = ', '.join(args)
		)
	
	@classmethod
	def callable_signature_to_string(cls, callable_obj, objname=None):
		sign = inspect.signature(callable_obj)
		objname = objname or cls.object_name(callable_obj)
		
		return cls.repr_with_args(
			objname, 
			args = [
				arg.name + ('?' if arg.default != arg.empty else '')
				for arg in sign.parameters.values()
				if arg.kind != arg.VAR_KEYWORD
			],
		)
	
	def __repr__(self):
		return self.callable_signature_to_string(self, self.object_name(self))

# This class forms a chain of lists
class Chain(list):
	def __init__(self, *transform_list):
		# for e.g: Chain([a1, a2, a3])
		if transform_list.__len__() == 1 and isinstance(transform_list[0], list):
			super().__init__(transform_list[0])
		# for e.g: Chain(a1, a2, a3)
		else:
			super().__init__(transform_list)

	def __call__(self, frame, **_):
		for tr in self:
			if tr is not None:
				frame.apply(tr)

	@staticmethod
	def transform_name(name):
		if isinstance(name, types.FunctionType) or isinstance(name, types.MethodType):
			return Base.callable_signature_to_string(name)
		elif isinstance(name, torch.nn.Module):
			return '<module> ' + Base.callable_signature_to_string(name)
		else:
			repr = str(name)
			if repr.__len__() > 87:
				repr =  repr[:87] + '...'

			repr = repr.replace('\n', ' ')
			return repr

	def __repr__(self):
		self_name = self.__class__.__name__

		if self:
			return self_name + '(\n	' + '\n	'.join(map(self.transform_name, self)) + '\n)'
		else:
			return self_name + '()'

	def __add__(self, other):
		return Chain(super().__add__(other))

	def copy(self):
		return Chain(super().copy())

#This class performs a transform that applies a 1 to 1 mapping on selected fields
class ByField(Base):
	# for e.g: ByField([(img1, img2)], torch.from_numpy)
	def __init__(self, fields='*', operation=None):
		"""
		:param fields: list of fields on which the function can be applied. The following can be taken as possible inputs:
			a string - frame[field] = func(frame[field])
			a tuple (in, out) - frame[out] = func(frame[in])
			fields=='*' selects all fields
		:param operation: operation to be applied on those fields
		"""
	
		if fields == '*':
		# for e.g : ByField('*')
			self.fields_all = True
		elif isinstance(fields, str):
		# for e.g : ByField('field')
			self.fields_all = False
			self.field_pairs = [(fields, fields)]
		elif isinstance(fields, dict):
		# for e.g : ByField({'a1in': 'a1out', 'a2in': 'a2out'})
			self.fields_all = False

			self.field_pairs = list(fields.items())
		else:
		# for e.g : ByField(['a1', (a1, a2)])
			self.fields_all = False

			self.field_pairs = [
				f if isinstance(f, tuple) or isinstance(f, list) else (f, f)
				for f in fields
			]
	
		self.operation = operation
	
	def forward(self, field_name, value):
		if self.operation is not None:
			return self.operation(value)
		else:
			raise NotImplementedError("Error in ByField::forward")
	
	def raise_error_about_type(self, field_name, value, should_be):
		if not self.fields_all:
			# only raise error if the field was specifically requested
			log.warning(f'{self} request for field {field_name} which is a {type(value)} and not a {should_be}')
	
	def __call__(self, frame, **_):
		if not self.fields_all:
			return {
				fi_out: self.forward(fi_in, frame[fi_in])
				for(fi_in, fi_out) in self.field_pairs
			}
		else:
			return {
				fi: self.forward(fi, val)
				for fi, val in frame.items()
			}

	def __repr__(self):
		return self.repr_with_args(
			name = '{name}{op}'.format(
				name = Base.object_name(self),
				op = ('<' + Base.object_name(self.operation) + '>') if self.operation is not None else '',
			),
			args = '*' if self.fields_all else [
				fp[0] if fp[0] == fp[1] else (fp[0] + ' -> ' + fp[1])
				for fp in self.field_pairs
			],
		)

# This class renames a specific field if needed
class RenameKw(Base):
	def __init__(self, *args, b_copy=False, **field_dict):
		if args.__len__() == 1:
			a0 = args[0]
			if isinstance(a0, list):
				self.field_pairs = a0
			else:
				self.field_pairs = args[0].items()

		elif args.__len__() == 0:
			self.field_pairs = field_dict.items()
		else:
			raise Exception("Incorrect format. Argument passed should be RenameKw({'a': 'b'}) or RenameKw(a = 'b')")

		self.b_copy = b_copy

	def __call__(self, frame, **_):
		return {
			new_name: frame[old_name] if self.b_copy else frame.pop(old_name)
			for (old_name, new_name) in self.field_pairs
		}

	def __repr__(self):
		return self.repr_with_args(
			self.object_name(self),
			(fp[0] + ' -> ' + fp[1] for fp in self.field_pairs),
		)

TrRename = RenameKw

#This class keeps only the required fields
class KeepFields(Base):
	def __init__(self, *fields):
		# For e.g : KeepFields([a1, a2, a3])
		if fields.__len__() == 1 and isinstance(fields[0], list):
			self.fields = set(fields[0])
		# For e.g : KeepFields(a1, a2, a3)
		else:
			self.fields = set(fields)

	def __call__(self, frame, **_):
		existing = set(frame.keys())
		missing = self.fields - existing
		if missing:
			log.error(f'Some fields are required. KeepFields: missing fields {missing}')

		to_remove = existing - self.fields

		for field in to_remove:
			del frame[field]
			
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), self.fields)

#This class keeps only the required fields given the prefix
class KeepFieldsByPrefix(Base):
	def __init__(self, *prefixes):
		# For e.g : KeepFieldsByPrefix([a1, a2, a3])
		if prefixes.__len__() == 1 and isinstance(prefixes[0], list):
			self.prefixes = prefixes[0]
		# For e.g : KeepFieldsByPrefix(a1, a2, a3)
		else:
			self.prefixes = prefixes
	
	def should_field_be_kept(self, field):
		for a in self.prefixes:
			if field.startswith(a):
				return True
		return False
	
	def __call__(self, frame, **_):
		to_keep = set(k for k in frame.keys() if self.should_field_be_kept(k))
		to_remove = set(frame.keys()) - to_keep
		for field in to_remove:
			del frame[field]
			
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), [p+'*' for p in self.prefixes])

#This class converts type1 to type2
class AsType(Base):
	def __init__(self, name_to_type):
		self.name_to_type = name_to_type

	@staticmethod
	def convert(val, type):
		if isinstance(val, np.ndarray):
			return val.astype(type)
		if isinstance(val, torch.Tensor):
			return val.type(type)
		raise NotImplementedError('Unsupported format. Neither an ndarray nor a torch.Tensor')

	def __call__(self, **fields):
		return {
			name: self.convert(fields[name], tp)
			for name, tp in self.name_to_type.items()
		}

	def __repr__(self):
		return self.repr_with_args(self.object_name(self), [
			'{n}->{t}'.format(n=name, t=tp)
			for name, tp in self.name_to_type.items()
		])

class FrameTransform:
	def __init__(self, func):
		self.func = func

	def __call__(self, frame):
		override_result = self.func(frame=frame, **frame)
		if override_result is not None:
			frame.update(override_result)
		return frame

	def __repr__(self):
		return Base.callable_signature_to_string(self.func, Base.object_name(self.func))

	#this function wraps the internal function
	def partial(self, *args, **kwargs):
		return FrameTransform(partial(self.func, *args, **kwargs))

class NChain(Chain):
	def __call__(self, frame):
		for fr in self:
			frame = fr(frame)

		return frame

image_mean_default = np.array([0.485, 0.456, 0.406])[None, None, :]
image_std_default = np.array([0.229, 0.224, 0.225])[None, None, :]

#standardizing images
def zero_center_img(image, means = image_mean_default, stds = image_std_default):
	"""Standardization is a data scaling technique that assumes that the distribution of the data is Gaussian and shifts
	 the distribution of the data to have a mean of zero and a standard deviation of one. Standardization of images is
	 achieved by subtracting the mean pixel value and dividing the result by the standard deviation of the pixel values.
	 The mean and standard deviation statistics can be calculated on the training dataset
    """
	image_float = image.astype(np.float32)
	image_float *= (1./255.)
	image_float -= means
	image_float /= stds
	return image_float

# undo standardizing images
def zero_center_img_undo(image, means = image_mean_default, stds = image_std_default):
	image_new = image * stds
	image_new += means
	image_new *= 255
	return image_new.astype(np.uint8)

class ZeroCenterBase(ByField):
	def __init__(self, fields='*', means = image_mean_default, stds = image_std_default):
		super().__init__(fields=fields)
		self.means = np.array(means, dtype=np.float32).reshape(-1)[None, None, :]
		self.stds = np.array(stds, dtype=np.float32).reshape(-1)[None, None, :]

	def func(self, image):
		raise NotImplementedError()

	def forward(self, field_name, value):
		if isinstance(value, np.ndarray):
			# checks if the value is not 8bit and that the shape is of the format [H, W, 3]
			if value.shape.__len__() == 3 and value.shape[2] == 3:
				return self.func(value)
			else:
				return value
		else:
			self.raise_error_about_type(field_name, value, 'np.ndarray')
			return value

class ZeroCenterImgs(ZeroCenterBase):
	def func(self, image):
		return zero_center_img(image, self.means, self.stds)

class ZeroCenterImgsUndo(ZeroCenterBase):
	def func(self, image):
		# if the value is 8bit, it cannot be zero centered
		if image.dtype != np.uint8:
			return zero_center_img_undo(image, self.means, self.stds)
		else:
			return image

#Horizontal flip only for numpy images
class RandomlyFlipHorizontal(Base):
	def __init__(self, fields=['image', 'labels']):
		self.fields = fields

	def __call__(self, **frame_values):
		if np.random.random() > 0.5:
			return {
				fi: frame_values[fi][:, ::-1].copy()
				for fi in self.fields
			}
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), self.fields)


class RandomCrop(Base):

	def __init__(self, crop_size=[512, 1024], fields=['image', 'labels']):
		self.fields = fields
		self.crop_size = np.array(crop_size, dtype=np.int)

	def __call__(self, **frame_values):
		shape = np.array(frame_values[self.fields[0]].shape[:2])

		if np.any(shape < self.crop_size):
			raise ValueError(
				'Image size is {sh} but cropping requested to be in size {cr}'.format(sh=shape, cr=self.crop_size))

		space = shape - self.crop_size
		top_left = np.array([np.random.randint(0, space[i]) for i in range(2)])
		bottom_right = top_left + self.crop_size

		return {
			fi: frame_values[fi][top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
			for fi in self.fields
		}


class Show(Base):
	def __init__(self, *channel_names):
		self.channel_names = channel_names

	@classmethod
	def get_channel_values(cls, fields, name_or_names):
		if isinstance(name_or_names, str):
			return fields.get(name_or_names, None)
		elif name_or_names is None:
			return None
		else:
			return list(map(partial(cls.get_channel_values, fields), name_or_names))

	def __call__(self, **fields):
		show(*self.get_channel_values(fields, self.channel_names))

#Important transform which sends the specified fields to GPU
class SendCUDA(ByField):
	def forward(self, field_name, val):
		if not torch.cuda.is_available():
			return val

		if isinstance(val, torch.Tensor):
			return val.cuda()
		elif isinstance(val, np.ndarray):
			return torch.from_numpy(val).cuda()
		else:
			self.raise_error_about_type(field_name, val, 'torch.Tensor')
			return val

#Important transform which retrieves the specified fields to GPU
class TrNP(ByField):
	def forward(self, field_name, val):
		if isinstance(val, torch.Tensor):
			if val.requires_grad:
				return val.detach().cpu().numpy()
			else:
				return val.cpu().numpy()
		else:
			self.raise_error_about_type(field_name, val, 'torch.Tensor')
			return val

#Changes shape from [H, W, 3] format to [3, H, W] format
def torch_images(**fields):
	res = dict()
	for field, value in fields.items():
		if isinstance(value, np.ndarray) and value.shape.__len__() == 3 and value.shape[2] == 3:
			image_tran = torch.from_numpy(value.transpose(2, 0, 1))
			image_tran.requires_grad = False
			res[field] = image_tran
	return res


def untorch_images(**fields):
	res = dict()
	for field, value in fields.items():
		if hasattr(value, 'shape') and value.shape.__len__() == 3 and value.shape[0] == 3:
			if isinstance(value, torch.Tensor):
				value = value.cpu().numpy()
			res[field] = value.transpose(1, 2, 0)
	return res

def torch_onehot(index, num_channels, dtype=torch.uint8):
	#any value more than num_channels will be converted to 0
	roi = index < num_channels
	index = index.byte().clone()
	index *= roi.byte()

	onehot = torch.zeros(
		index.shape[:1] + (num_channels,) + index.shape[1:],
		device=index.device,
		dtype=dtype,
	)
	onehot.scatter_(1, index[:, None].long(), roi[:, None].type(dtype))
	return onehot

class ChannelIOBase(Base):
	def __init__(self, channel, field):
		self.field_name = field
		self.channel = channel
		self.channel_from_frame = isinstance(self.channel, str)

	def get_channel(self, dset_from_frame):
		if self.channel_from_frame:
			if dset_from_frame is None:
				raise ValueError(f"""Tried loading channel {self.channel} from specified dataset but frame has no "dset" field. 
Transform: {self}.""")
			return dset_from_frame.channels[self.channel]
		else:
			return self.channel

	def __repr__(self):
		return f'{self.__class__.__name__}({self.field_name} ~ channel {self.channel})'

class ChannelLoad(ChannelIOBase):
	def __call__(self, frame, dset=None, **_):
		self.get_channel(dset).load(dset, frame, self.field_name)

class ChannelSave(ChannelIOBase):

	def __call__(self, frame, dset=None, **_):
		self.get_channel(dset).save(dset, frame, self.field_name)