"""
DESCRIPTION:     Python script for parallelizing the entire pipeline
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""


import logging
log = logging.getLogger('exp')
from .frame import *
from .transforms import Chain
from ..pipeline.utils import ProgressBar
from torch.utils.data.dataloader import DataLoader
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.utils.data
from torch.utils.data._utils.collate import string_classes, int_classes, np_str_obj_array_pattern
import collections
import gc  #garbage collector

#Puts each data field into a tensor with outer dimension batch size
def default_collate_edited(batch):
	elem = batch[0]
	elem_type = type(elem)
	if isinstance(elem, torch.Tensor):
		out = None
		if torch.utils.data.dataloader is not None:
			# Concatenate directly into a shared memory tensor to avoid an extra copy
			numel = sum([x.numel() for x in batch])
			storage = elem.storage()._new_shared(numel)
			out = elem.new(storage)
		return torch.stack(batch, 0, out=out)
	elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
		if elem_type.__name__ == 'ndarray':
			# array of string classes and object
			if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
				raise TypeError(f'Incorrect numpy dtype {elem.dtype}')

			return default_collate_edited([torch.as_tensor(b) for b in batch])
		elif elem.shape == ():
			return torch.as_tensor(batch)
	elif isinstance(elem, float):
		return torch.tensor(batch, dtype=torch.float64)
	elif isinstance(elem, int_classes):
		return torch.tensor(batch)
	elif isinstance(elem, string_classes):
		return batch
	elif isinstance(elem, collections.Mapping):
		return {key: default_collate_edited([d[key] for d in batch]) for key in elem}
	elif isinstance(elem, collections.Sequence):
		transposed = zip(*batch)
		return [default_collate_edited(samples) for samples in transposed]
	return batch

class Pipeline:
	#change default arguments based on memory
	default_loader_args = dict(
		shuffle=False,
		batch_size = 1,  #4
		num_workers = 0,  #0
		drop_last = False,
	)

	def __init__(self, tr_input=None, batch_pre_merge=None, tr_batch=None, tr_output=None, loader_args=default_loader_args):
		self.tr_input = tr_input or Chain()
		self.batch_pre_merge = batch_pre_merge or Chain()
		self.tr_batch = tr_batch or Chain()
		self.tr_output = tr_output or Chain()

		self.loader_class = DataLoader
		self.loader_args = loader_args

	def __repr__(self):
		return ('Pipeline(\n' + '\n'.join(
			f + ' = ' + str(v)
			for (f, v) in self.__dict__.items() if f.startswith('tr_')
		) + '\n)')


	@staticmethod
	#Call necessary transforms
	def pipeline_collate(tr_input, batch_pre_merge, frames):
		result_frame = Frame()
		to_batch = []
		for fr in frames:
			fr.apply(tr_input)
			fr_to_batch = fr.copy()
			fr_to_batch.apply(batch_pre_merge)
			to_batch.append(fr_to_batch)

		result_frame = Frame(default_collate_edited(to_batch))
		return dict(batch=result_frame, input_frames = frames)

	@staticmethod
	#Split 0-dim tensors
	def unbatch_value(value, idx):
		if not hasattr(value, "__getitem__"):
			return value
		elif isinstance(value, torch.Tensor) and value.shape.__len__() == 0:
			return value
		else:
			return value[idx]
	
	@classmethod
	def unbatch(cls, batch, input_frames):
		return [
			in_fr.copy().update({
				field: cls.unbatch_value(value, idx) for field, value in batch.items()
			})
			for (idx, in_fr) in enumerate(input_frames)
		]


	def get_loader(self, dataset):
		collate_fn = partial(self.pipeline_collate, self.tr_input, self.batch_pre_merge)
		return self.loader_class(dataset, collate_fn=collate_fn, **self.loader_args)

	def apply_tr_output(self, frame):
		frame.apply(self.tr_output)
		return frame

	def progress(self, dset, b_accumulate=True, b_grad=False, b_one_batch=False, b_pbar=True, log_progress_interval=None, short_epoch=None):
		loader = self.get_loader(dset)

		out_all_frames = [] if b_accumulate else None

		if b_pbar and (not b_one_batch):
			pbar = ProgressBar(dset.__len__())
		else:
			pbar = 0

		frames_proc = 0
		batches_proc = 0

		with ThreadPoolExecutor(max_workers=max(self.loader_args['batch_size'], 1)) as pool:
			#print("max_workers",max_workers)
			with torch.set_grad_enabled(b_grad):
				for loader_item in loader:
					batch = loader_item['batch']
					input_frames = loader_item['input_frames']
					batch.apply(self.tr_batch)
					out_frames = self.unbatch(batch, input_frames)

					# parallelize all outputs
					out_frames = list(pool.map(self.apply_tr_output, out_frames))
					if b_one_batch:
						return batch, out_frames

					del batch

					if b_accumulate:
						out_all_frames += out_frames

					# display progress bar
					frames_proc += out_frames.__len__()
					pbar += out_frames.__len__()

					if log_progress_interval and frames_proc % log_progress_interval < out_frames.__len__():
						frame_num = dset.__len__()
						if short_epoch is not None: frame_num = min(short_epoch, frame_num)
						log.debug('	{p}	/	{t}'.format(p=frames_proc, t=frame_num))

					if short_epoch is not None and frames_proc >= short_epoch:
						return out_all_frames

					# perform GC periodically
					batches_proc += 1
					if batches_proc % 8 == 0:
						gc.collect()
		return out_all_frames