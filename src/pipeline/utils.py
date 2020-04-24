"""
DESCRIPTION:     Python script for helper functions for creating output images
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

Code for Context manager for changing directory was taken from http://ralsina.me/weblog/posts/BB963.html
Date : 20.10.2019
"""

import PIL.Image
from matplotlib import cm
import matplotlib as mpl
from io import BytesIO
from binascii import b2a_base64
from IPython.display import display_html
import h5py, os
from torch.nn import functional as torch_functional
from contextlib import contextmanager
from pathlib import Path
import numpy as np
mpl.use('Agg')
from tqdm import tqdm
np.set_printoptions(suppress=True, linewidth=180)

try:
	import cv2
except Exception as e:
	print('OpenCV import fail:', e)

try:
	import torch
	def ensure_numpy_image(img):
		if isinstance(img, torch.Tensor):
			img = img.cpu().numpy().transpose([1, 2, 0])
		return img
except:
	def ensure_numpy_image(img):
		return img

def adapt_to_img_data(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed()):
	num_dimensions = img_data.shape.__len__()

	if num_dimensions == 3:
		if img_data.dtype != np.uint8:
			if np.max(img_data) < 1.1:
				img_data = img_data * 255
			img_data = img_data.astype(np.uint8)

	elif num_dimensions == 2:
		if img_data.dtype == np.bool:
			img_data = img_data.astype(np.uint8)*255

		else:
			v_max = np.max(img_data)
			if img_data.dtype == np.uint8 and v_max == 1:
				img_data = img_data * 255

			else:
				v_min = np.min(img_data)

				if v_min >= 0:
					img_data = (img_data - v_min) * (1 / (v_max - v_min))
					img_data = cmap_pos(img_data, bytes=True)[:, :, :3]

				else:
					v_range = max(-v_min, v_max)
					img_data = img_data / (2 * v_range) + 0.5
					img_data = cmap_div(img_data, bytes=True)[:, :, :3]

	return img_data

class ImageHTML:
    CONTENT_TMPL = """<div style="width:100%;"><img src="data:image/{fmt};base64,{data}" /></div>"""

    def __init__(self, image, fmt='webp', adapt=True):
        self.fmt = fmt
        image = adapt_to_img_data(image) if adapt else image
        self.data_base64 = self.encode_image(image, fmt)

    @staticmethod
    def encode_image(image, fmt):
        with BytesIO() as buffer:
            PIL.Image.fromarray(image).save(buffer, format=fmt)
            image_base64 = str(b2a_base64(buffer.getvalue()), 'utf8')
        return image_base64

    def _repr_html_(self):
        return self.CONTENT_TMPL.format(fmt=self.fmt, data=self.data_base64)

    def show(self):
        display_html(self)

class ImageGridHTML:
    start_row = """<div style="display:flex; justify-content: space-evenly;">"""
    end_row = """</div>"""

    def __init__(self, *rows, fmt='webp', adapt=True):
        """
        :param fmt: Available image formats: "png", "jpeg", "webp"
        :param adapt: Specify if conversion of unusual shapes and datatypes to the RGB format is needed

        For e.g: show(img_1, img_2) will display each image on a separate row
        For e.g: show([img_1, img_2])` will display both images in a single row
        For e.g: show([img_1, img_2], [img_3, img_4])` will display images in two rows
        """
        self.fmt = fmt
        self.adapt = adapt
        self.rows = [self.encode_row(r) for r in rows]

    def encode_row(self, row):
        if isinstance(row, (list, tuple)):
            return [ImageHTML(img, fmt=self.fmt, adapt=self.adapt) for img in row if img is not None]
        elif row is None:
            return []
        else:
            return [ImageHTML(row, fmt=self.fmt, adapt=self.adapt)]

    def _repr_html_(self):
        regions = []

        for row in self.rows:
            regions.append(self.start_row)
            regions += [img._repr_html_() for img in row]
            regions.append(self.end_row)

        return '\n'.join(regions)

    def show(self):
        display_html(self)

    @staticmethod
    def show_image(*images, **options):
        """
        :param fmt: Available image formats: "png", "jpeg", "webp"
        :param adapt: Specify if conversion of unusual shapes and datatypes to the RGB format is needed

        For e.g: show(img_1, img_2) will display each image on a separate row
        For e.g: show([img_1, img_2])` will display both images in a single row
        For e.g: show([img_1, img_2], [img_3, img_4])` will display images in two rows
        """
        ImageGridHTML(*images, **options).show()


show = ImageGridHTML.show_image

# Context manager for changing directory was taken from http://ralsina.me/weblog/posts/BB963.html
@contextmanager
def current_dir(new_cwd):
	old_cwd = Path.cwd()
	os.chdir(new_cwd)
	try:
		yield
	finally:
		os.chdir(old_cwd)

def convert_img_to_display(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed()):
	num_dimensions = img_data.shape.__len__()

	if num_dimensions == 3:
		if img_data.shape[2] > 3:
			img_data = img_data[:, :, :3]

		if img_data.dtype != np.uint8 and np.max(img_data) < 1.1:
			img_data = (img_data * 255).astype(np.uint8)


	elif num_dimensions == 2:
		if img_data.dtype == np.bool:

			img_data = img_data.astype(np.uint8)*255
			img_data = np.stack([img_data]*3, axis=2)

		else:
			v_max = np.max(img_data)
			if img_data.dtype == np.uint8 and v_max == 1:
				img_data = img_data * 255

			else:
				v_min = np.min(img_data)

				if v_min >= 0:
					img_data = (img_data - v_min) * (1 / (v_max - v_min))
					img_data = cmap_pos(img_data, bytes=True)[:, :, :3]

				else:
					v_range = max(-v_min, v_max)
					img_data = img_data / (2 * v_range) + 0.5
					img_data = cmap_div(img_data, bytes=True)[:, :, :3]

	return img_data

# Saving hdf5 files
def hdf5_save(path, data):
	with h5py.File(path, 'w') as fout:
		for name, value in data.items():

			if isinstance(value, np.ndarray) and np.prod(value.shape) > 1 and value.dtype == np.bool:
				fout.create_dataset(name, data=value, compression=7)
			else:
				fout[name] = value

# Loading hdf5 files
def hdf5_load(path):
	with h5py.File(path, 'r') as fin:
		result = dict()

		def visit(name, obj):
			if isinstance(obj, h5py.Dataset):
				result[name] = obj[()]

		fin.visititems(visit)

		return result

class Padder:
#Image is padded here so that the size is divisible by the argument `divisor`
# Final result is unpadded later

    def __init__(self, shape, divisor):
        size = np.array(shape[-2:])
        size_deficit = np.mod(divisor - np.mod(size, divisor), divisor)

        self.needs_padding = np.any(size_deficit)

        self.size_orig = size
        self.padding_tl = size_deficit // 2
        self.padding_br = (size_deficit + 1) // 2
        self.unpad_end = self.padding_tl + self.size_orig

    def pad(self, tensor):
        if self.needs_padding:

            # the padding process needs to have the format [B, C, H, W], so make sure that C dimension is present even if its just 1. Remove it later
            expand_chanels = tensor.shape.__len__() == 3
            if expand_chanels:
                tensor = torch.unsqueeze(tensor, 1)

            # since pytorch cannot perform reflect for uint8, use constant
            mode = 'reflect' if tensor.dtype != torch.uint8 else 'constant'

            res = torch_functional.pad(
                tensor,
                (self.padding_tl[1], self.padding_br[1], self.padding_tl[0], self.padding_br[0]),
                mode=mode,
            )
            #print(res)
            # Remove C dimension
            if expand_chanels:
                res = torch.squeeze(res, 1)
            return res
        else:
            return tensor

    def unpad(self, tensor):
        if self.needs_padding:
            if tensor.shape.__len__() == 4:
                res = tensor[:, :, self.padding_tl[0]:self.unpad_end[0], self.padding_tl[1]:self.unpad_end[1]]
            elif tensor.shape.__len__() == 3:
                res = tensor[:, self.padding_tl[0]:self.unpad_end[0], self.padding_tl[1]:self.unpad_end[1]]
            else:
                raise ValueError(f'Unpadding takes a 3 or 4 dim value, but got shape {tuple(tensor.shape)}')
            return res
        else:
            return tensor

# Good to see the progress occasionally
class ProgressBar:
	def __init__(self, goal):
		self.goal = goal
		self.value = 0
		self.bar = tqdm(total=goal)

	def __iadd__(self, change):
		self.value += change
		self.bar.update(change)

		if self.value >= self.goal:
			self.bar.close()

		return self


class bind:
    """
    For e.g :   bind(func, labels='labels_trainId').outs(partitions='labels_partition') will produce the output in
                r = func(labels = input['labels_trainId'])
                output['labels_partition'] = r['partitions']

    For e.g :  bind(torch.sigmoid, 'pred_logits').outs('pred_probs')

    For e.g :  bind(torch.sigmoid).ins('pred_logits').outs('pred_probs')
    """
    out_single_name: str = None
    in_single_name: str = None
    output_binds = None
    input_binds = None
    default_args = {}

    @staticmethod
    def args_kwargs_to_bindings(args, kwargs):
        return dict(
            **kwargs,
            **{
                a: a for a in args
            }
        )

    def __init__(self, func, *args, **kwargs):

        self.func = func
        if args.__len__() == 1 and not kwargs:
            self.in_single_name = args[0]
        else:
            self.input_binds = self.args_kwargs_to_bindings(args, kwargs)

    def outs(self, *args, **kwargs):

        if args.__len__() == 1 and not kwargs:
            self.out_single_name = args[0]
        else:
            self.output_binds = self.args_kwargs_to_bindings(args, kwargs)

        return self

    def defaults(self, **kwargs):
        self.default_args = kwargs
        return self

    def __call__(self, **fields):
        if self.in_single_name:
            result = self.func(fields[self.in_single_name], **self.defaults)
        else:
            result = self.func(**{
                arg_name: fields[named_field]
                for (arg_name, named_field) in self.input_binds.items()
            }, **self.default_args)

        if self.out_single_name:
            return {self.out_single_name: result}

        elif result is not None:
            if not isinstance(result, dict):
                raise ValueError(
                    f'Function return is not a dict but {type(result)}. There is no single output name, tr={self}')

            # If there are no bindings, just return names
            if self.output_binds is None:
                return result

            return {
                bound_out_name: result[func_out_name]
                for (func_out_name, bound_out_name) in self.output_binds.items()
            }

    def __repr__(self):
        return f'bind({self.func})'
