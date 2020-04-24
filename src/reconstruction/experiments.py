"""
DESCRIPTION:     Python script for creating pipeline for pix2pixHD
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

A portion of this code is based on https://github.com/NVIDIA/pix2pixHD which is licensed
under the BSD License.
Date: 20.10.2019

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
All rights reserved.

For details on the license please have a look at MasterThesis/Licenses/pix2pixHD_License.txt
"""

import cv2, torch
import numpy as np
from ..paths import exp_dir
from ..pipeline.pipeline import Pipeline
from ..pipeline.transforms import Chain, KeepFields,torch_onehot, SendCUDA, TrNP
from ..datasets.cityscapes import CityscapesLabelInfo

class Pix2PixHD_GAN:

	pix2pixHD_variants = {

		'pix2pixHD': f"""
			--name pix2pixHD_model            
			--checkpoints_dir {exp_dir}
			--no_instance
			--resize_or_crop crop --fineSize 384 --batchSize 4
		""",
	}

	NET_CACHE = dict()
	#change arguments based on memory
	loader_args = dict(
		shuffle = False,
		batch_size = 4,
		num_workers = 0,
		drop_last = False,
	)

	@classmethod
	def load_pix2pixHD(cls, variant='pix2pixHD'):
		print("Loading Pix2PixHD model")
		#not implemented
		if variant in cls.NET_CACHE:
			return cls.NET_CACHE[variant]
		from pjval_ml.OSR.GAN.src.reconstruction.pix2pixHD.options.test_options import TestOptions
		from pjval_ml.OSR.GAN.src.reconstruction.pix2pixHD.models.models import create_model

		"""This code is based on https://github.com/NVIDIA/pix2pixHD which is licensed under the BSD License."""

		opt = TestOptions().parse(save=False, override_args=cls.pix2pixHD_variants[variant].split())

		#pix2pixHD supports nThreads = 1 and batchSize = 1
		opt.nThreads = 1
		opt.batchSize = 1
		opt.serial_batches = True  # do not shuffle
		opt.no_flip = True  # do not flip
		pix2pixHD_module = create_model(opt)
		cls.NET_CACHE[variant] = pix2pixHD_module
		return pix2pixHD_module


	def __init__(self, variant='pix2pixHD'):
		self.module_pix2pix = self.load_pix2pixHD(variant)
		self.tabulate_trainId_to_fullId_cuda = torch.from_numpy(CityscapesLabelInfo.tabulate_trainId_to_label).byte().cuda()

	@staticmethod
	def untorch_image(recon_image, **_):
		return dict(recon_image = recon_image.transpose([1, 2, 0]))

	def tr_gan(self, pred_labels_ID, **_):
		labels = self.tabulate_trainId_to_fullId_cuda[pred_labels_ID.reshape(-1).long()].reshape(pred_labels_ID.shape)
		one_hot_labels = torch_onehot(
			labels,
			num_channels = self.module_pix2pix.opt.label_nc,
			dtype = torch.float32,
		)

		inst = None
		img = None
		recon_out = self.module_pix2pix.inference(one_hot_labels, inst, img)

		desired_shape = one_hot_labels.shape[2:]
		if recon_out.shape[2:] != desired_shape:
			recon_out = recon_out[:, :, :desired_shape[0], :desired_shape[1]]

		recon_image = (recon_out + 1) * 128
		recon_image = torch.clamp(recon_image, min=0, max=255)
		recon_image = recon_image.type(torch.uint8)

		return dict(
			recon_image_raw = recon_out,
			recon_image = recon_image,
		)
	#using only the reconstructed image
	def tr_gan_np(self, pred_labels_ID, **_):
		pred_labels_ID = torch.from_numpy(pred_labels_ID)
		pred_labels_ID = pred_labels_ID[None]
		out = self.tr_gan(pred_labels_ID=pred_labels_ID.cuda())
		recon_image = out['recon_image'][0].cpu()

		recon_image = recon_image.numpy().transpose(1, 2, 0)
		return dict(
			recon_image=recon_image,
		)
	#constructing pipeline for reconstructing image using pix2pixHD
	def construct_pix2pixHD_pipeline(self):
		return Pipeline(
			tr_input = Chain(
			),
			batch_pre_merge= Chain(),
			tr_batch = Chain(
				SendCUDA(),
				self.tr_gan,
				KeepFields('recon_image'),

			),
			tr_output = Chain(
				TrNP(),
				self.untorch_image,
			),
			loader_args = self.loader_args,
		)
#get the required class names form the pixels
def instances_from_semantics(labels_source, min_size=None, allowed_classes=None, forbidden_classes=None, **_):
	set_label = set(np.unique(labels_source))

	if allowed_classes is not None:
		set_label = set_label.intersection(set(allowed_classes))

	if forbidden_classes is not None:
		set_label = set_label.difference(set(forbidden_classes))

	set_label = list(set_label)
	set_label.sort()

	inst_idx = 1
	out_instance_map = np.zeros(labels_source.shape, dtype=np.int32)

	for a in set_label:
		curr_label_map = labels_source == a
		num_pix = np.count_nonzero(curr_label_map)

		if num_pix > 0:
			num_cc, cc_map = cv2.connectedComponents(curr_label_map.astype(np.uint8))

			for idx in range(1, num_cc):
				instance_mask = cc_map == idx

				if (min_size is None) or np.count_nonzero(instance_mask) > min_size:
					out_instance_map[instance_mask] = inst_idx
					inst_idx += 1

	return dict(
		instances=out_instance_map,
	)