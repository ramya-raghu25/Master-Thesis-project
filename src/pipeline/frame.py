"""
DESCRIPTION:     Python script for multiprocessing functions
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

import logging
log = logging.getLogger('exp')
import multiprocessing, multiprocessing.dummy
from functools import partial
from ..pipeline.utils import ProgressBar

class Frame(dict):
	def __getattr__(self, key):
		# not called
		try:
			return self[key]
		except KeyError as e:
			raise AttributeError(e)

	def __setattr__(self, key, value):
		self[key] = value

	#Update data with values returned by function as kwargs
	def apply(self, f):
		try:
			fr = f(frame=self, **self)
			if fr:
				self.update(fr)
			return fr
			
		except KeyError as e:
			log.error('Missing field {field} for transform {t}'.format(field=e, t=f))

	def __call__(self, f):
		return self.apply(f)

	def copy(self):
		return Frame(super().copy())

	def update(self, *args, **kwargs):
		super().update(*args, **kwargs)
		return self

	@staticmethod
	def repr_field(val):
		if isinstance(val, dict):
			return '{' + ''.join(f'\n		{i}: {Frame.repr_field(j)}'  for i, j in val.items()) + '}'

		if isinstance(val, list):
			val_show = val[:10] if val.__len__() > 10 else val
			return '[' + '\n'.join(map(Frame.repr_field, val_show)) + ']'

		val_shape = getattr(val, 'shape', None)
		val_dtype = getattr(val, 'dtype', None)
		if not (val_shape is None or val_dtype is None):
			if val_shape.__len__() > 0:
				return '{tp}[{dims}]'.format(dims = ', '.join(map(str, val_shape)),tp = val_dtype,
				)

		return str(val)


	def __repr__(self):
		return ''.join(
			["Frame(\n"]+[	'	{fn} = {fval}\n'.format(fn=fn, fval = self.repr_field(fval))
			for fn, fval in self.items()
			]	+	[')'])

	@staticmethod
	def frame_worker(func, fr):
		fr.apply(func)
		return fr

	@staticmethod
	def build_pool(n_proc, n_threads):
		if n_proc > 1:
			return multiprocessing.Pool(n_proc)
		if n_threads > 1:
			return multiprocessing.dummy.Pool(n_threads)

	@classmethod
	def frame_listapply(cls, func, frames, n_proc=1, n_threads=1, batch = 2, ret_frames=False, pbar=True):
		if ret_frames:
			out_frames = []

		if pbar:
			pbar = ProgressBar(frames.__len__())
		else:
			pbar = 0

		if n_proc == 1 and n_threads == 1:
			for f in frames:
				f.apply(func)
				pbar += 1

				if ret_frames:
					out_frames.append(f)
		else:
			task = partial(cls.frame_worker, func)

			with cls.build_pool(n_proc, n_threads) as pool:
				for fr in pool.imap(task, frames, chunksize=batch):
					pbar += 1
					
					if ret_frames:
						out_frames.append(fr)

		if ret_frames:
			return out_frames

	# @classmethod
	# def parallel_process(cls, func, frames, n_proc=1, n_threads=4, batch = 4, ret_frames=True, pbar=True):
	# 	if ret_frames:
	# 		out_frames = []
	#
	# 	if pbar:
	# 		pbar = ProgressBar(frames.__len__())
	# 	else:
	# 		pbar = 0
	#
	# 	if n_proc == 1 and n_threads == 1:
	# 		for f in frames:
	# 			f = func(f)
	# 			pbar += 1
	#
	# 			if ret_frames:
	# 				out_frames.append(f)
	# 	else:
	# 		with cls.build_pool(n_proc, n_threads) as pool:
	# 			for fr in pool.imap(func, frames, chunksize=batch):
	# 				pbar += 1
	# 				if ret_frames:
	# 					out_frames.append(fr)
	#
	# 	if ret_frames:
	# 		return out_frames