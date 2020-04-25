"""
DESCRIPTION:     Python file for creating Unknown Object Segmentation Network
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

This code is taken and modified from https://github.com/ohosseini/DOCS-pytorch which is licensed
under the MIT License.
Copyright (c) 2018 Lynton Ardizzone
Date: 20.10.2019

A portion of this code is also based on https://github.com/zijundeng/pytorch-semantic-segmentation which is licensed
under the MIT License.
Copyright (c) 2017 ZijunDeng
Date: 20.10.2019


For details on the license please have a look at MasterThesis/Licenses/MIT_License.txt
"""

import numpy as np
import torch, os
import logging
log = logging.getLogger('exp')
from torch import nn
from ..pipeline.transforms import torch_onehot
from ..pipeline.utils import Padder
from pjval_ml import PJVAL_SHARE

class VGG(nn.Module):

	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	"""This code is taken and modified from https://github.com/ohosseini/DOCS-pytorch which is licensed
     under the MIT License."""

	def _initialize_weights(self):  #from docs.py
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):   #from DOCS.py
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfgs = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
	if pretrained:
		state_dict = torch.load(os.path.join(PJVAL_SHARE, 'Master_Thesis_Ramya', 'datasets', 'Weights', 'vgg16-397923af.pth'))
#Z:\Master_Thesis_Ramya\datasets\Weights
		model.load_state_dict(state_dict)
	return model


def vgg16(pretrained=False, progress=True, **kwargs):
	return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

class CorrelationOp(nn.Module):

	@staticmethod
	def operation(a, b):
		"""
		finds correlation between original image and reconstructed image using dot product
		B x C x H x W

		"""
		return torch.sum(a * b, dim=1, keepdim=True)

	def forward(self, a, b):
		return self.operation(a, b)

#VGG feature extractor for all 3 variants
class VggFeaturesExtract(nn.Module):
	VGG16_LAYERS = [3, 8, 15, 22, 29]

	def __init__(self, vgg_module, layers_to_extract, freeze=True):
		super().__init__()

		vgg_features = vgg_module.features

		ends = np.array(layers_to_extract, dtype=np.int) + 1
		starts = [0] + list(ends[:-1])

		self.slices = nn.Sequential(*[
			nn.Sequential(*vgg_features[s:e])
			for (s, e) in zip(starts, ends)
		])

		if freeze:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, image, **_):
		res = []
		value = image
		for i in self.slices:
			value = i(value)
			res.append(value)

		return res

###################################### Original image vs Reconstructed image 516##################################################
class OriginalVsReconstructed(nn.Module):
	#Has access only to Original image and Reconstructed image
	#decoder architecture
	#can also toggle between ReLU and SELU
	class UpBlock(nn.Sequential):
		def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True):

			modules = [
				nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
				nn.SELU(inplace=True),
				#nn.ReLU(True),
				nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
				nn.SELU(inplace=True),
				#nn.ReLU(True),
			]

			if b_upsample:
				modules += [
					nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
				]

			super().__init__(*modules)

	class CatCorr(nn.Module):
		def __init__(self, in_ch):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch*2, in_ch, kernel_size=1)

		def forward(self, previous, orig_feats, recon_feats):
			#this top layer comes from the previous layer of the upsampling pyramid
			channels = [previous] if previous is not None else []
			#sending correlated features and concatenated features from 2 streams to each decoder block
			channels += [
				#result of the correlation (dot product)  - it is a single channel with width=1
				CorrelationOp.operation(orig_feats, recon_feats),
				#performing 1x1 convolution which concatenates original image features and reconstructed image features
				self.conv_1x1(torch.cat([orig_feats, recon_feats], 1)),
			]
			return torch.cat(channels, 1)


	def __init__(self, num_outputs=2, freeze=True):
		super().__init__()
		vgg = vgg16(pretrained=True)
		self.vgg_extractor = VggFeaturesExtract(vgg_module= vgg,
			layers_to_extract =VggFeaturesExtract.VGG16_LAYERS[:4],
			freeze = freeze,
		)

		feature_channels = [512, 256, 128, 64]
		output_channels = [256, 256, 128, 64]
		previous_channels = [0] + output_channels[:-1]
		cmis, decs = [] , []

		for i, fc, oc, pc in zip(range(feature_channels.__len__(), 0, -1), feature_channels, output_channels, previous_channels):
			corr = self.CatCorr(fc)
			decoder = self.UpBlock(fc+1+pc, oc, oc, b_upsample=(i != 1))
			#print("corr",corr)  #prints n/w structure
			#print("decoder",decoder)  #prints n/w structure
			cmis.append(corr)
			decs.append(decoder)
		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(output_channels[-1], num_outputs, kernel_size=1)

	def forward(self, image, recon_image, **_):
		if recon_image.shape != image.shape:
			recon_image = recon_image[:, :, :image.shape[2], :image.shape[3]]

		if not self.training:
			Pad = Padder(image.shape, 16)
			image, recon_image = (Pad.pad(i) for i in (image, recon_image))
		#extracting features from original image and reconstructed image using VGG16   #516 step3
		vgg_img_feats = self.vgg_extractor(image)
		vgg_recon_feats = self.vgg_extractor(recon_image)

		value = None
		n_steps = self.cmis.__len__()

		for j in range(n_steps):
			i_inv = n_steps-(j+1)
			value = self.decs[j](
				self.cmis[j](value, vgg_img_feats[i_inv], vgg_recon_feats[i_inv])
			)

		res = self.final(value)

		if not self.training:
			res = Pad.unpad(res)
		return res

##################################################### Original image vs semantic label ######################
class OriginalVsLabel(nn.Module):
	#Has access only to Original image and semantic labels
	class Cat(nn.Module):
		def __init__(self, in_ch_img, in_ch_sem, out_ch):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch_img + in_ch_sem, out_ch, kernel_size=1)

		def forward(self, previous, orig_feats, sem_feats):
			#this top layer comes from the previous layer of the upsampling pyramid
			channels = [previous] if previous is not None else []

			#sending only concatenated features to each decoder block
			channels += [
				# performing 1x1 convolution which concatenates original image features and semantic label features
				self.conv_1x1(torch.cat([orig_feats, sem_feats], 1)),
			]
			return torch.cat(channels, 1)

	class SemFeatures(nn.Sequential):
		#Label features are extracted in OriginalVsLabel and OriginalVsReconstructedAndLabel
		def __init__(self, num_classes, feature_channels_sem):
			self.num_classes = num_classes

			layers = [nn.Sequential(
				nn.ReflectionPad2d(3),
				nn.Conv2d(num_classes, feature_channels_sem[0], kernel_size=7, padding=0),
				nn.ReLU(True),
			)]

			num_previous_ch = feature_channels_sem[0]
			for num in feature_channels_sem[1:]:
				layers.append(nn.Sequential(
					nn.Conv2d(num_previous_ch, num, kernel_size=3, stride=2, padding=1),
					nn.ReLU(True),
				))
				num_previous_ch = num

			super().__init__(*layers)

		def forward(self, labels):
			res = []
			value = torch_onehot(labels, self.num_classes, dtype=torch.float32)
			for i in self:
				value = i(value)
				res.append(value)
			return res

	def __init__(self, num_outputs=1, num_classes=19, freeze=True):
		super().__init__()
		self.num_classes = num_classes
		vgg = vgg16(pretrained=True)
		self.vgg_extractor = VggFeaturesExtract(vgg_module= vgg, layers_to_extract =VggFeaturesExtract.VGG16_LAYERS[:4],
			freeze = freeze,
		)

		feature_channels_vgg = [512, 256, 128, 64]
		feature_channels_sem = [256, 128, 64, 32]

		self.sem_extractor = self.SemFeatures(num_classes, feature_channels_sem[::-1])

		output_channels = [256, 256, 128, 64]
		previous_channels = [0] + output_channels[:-1]
		cmis, decs = [] , []

		for i, fc, sc, oc, pc in zip(range(feature_channels_vgg.__len__(), 0, -1), feature_channels_vgg,
				feature_channels_sem,
				output_channels,
				previous_channels
		):

			corr = self.Cat(fc, sc, fc)
			decoder = OriginalVsReconstructed.UpBlock(fc+pc, oc, oc, b_upsample=(i != 1))

			cmis.append(corr)
			decs.append(decoder)

		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(output_channels[-1], num_outputs, kernel_size=1)

	def forward(self, labels, image, **_):

		if not self.training:
			Pad = Padder(image.shape, 16)
			image, labels = (Pad.pad(i) for i in (image, labels))

		#extracting features from original image and semantic labels using VGG16
		vgg_img_feats = self.vgg_extractor(image)
		sem_feats = self.sem_extractor(labels)

		value = None
		n_steps = self.cmis.__len__()

		for j in range(n_steps):
			i_inv = n_steps-(j+1)
			value = self.decs[j](self.cmis[j](value, vgg_img_feats[i_inv], sem_feats[i_inv])
			)

		res = self.final(value)

		if not self.training:
			res = Pad.unpad(res)
		return res

########################################## Original image vs Reconstructed image and semantic labels ######################
class OriginalVsReconstructedAndLabel(nn.Module):
	#Has access to Original image, reconstructed image and semantic labels

	class CatCorrAll(nn.Module):
		def __init__(self, in_ch_img, in_ch_sem, ch_out):
			super().__init__()
			self.conv_1x1 = nn.Conv2d(in_ch_img*2 + in_ch_sem, ch_out, kernel_size=1)

		def forward(self, previous, orig_feats, recon_feats, sem_feats):
			#this top layer comes from the previous layer of the upsampling pyramid
			channels = [previous] if previous is not None else []

			#sending correlated features and concatenated features from 3 streams to each decoder block
			channels += [
				# result of the correlation (dot product)  - it is a single channel with width=1
				CorrelationOp.operation(orig_feats, recon_feats),
				# performing 1x1 convolution which concatenates original image features, reconstructed image features and semantic label features
				self.conv_1x1(torch.cat([orig_feats, recon_feats, sem_feats], 1)),
			]
			return torch.cat(channels, 1)

	def __init__(self, num_outputs=2, num_classes=19, freeze=True):
		super().__init__()

		self.num_classes = num_classes
		vgg = vgg16(pretrained=True)
		self.vgg_extractor = VggFeaturesExtract(vgg_module= vgg,	layers_to_extract =VggFeaturesExtract.VGG16_LAYERS[:4],
			freeze = freeze,
		)

		feature_channels_vgg = [512, 256, 128, 64]
		feature_channels_sem = [256, 128, 64, 32]

		self.sem_extractor = OriginalVsLabel.SemFeatures(num_classes, feature_channels_sem[::-1])

		output_channels = [256, 256, 128, 64]
		previous_channels = [0] + output_channels[:-1]

		cmis, decs = [] , []
		for i, fc, sc, oc, pc in zip(	range(feature_channels_vgg.__len__(), 0, -1),	feature_channels_vgg,
				feature_channels_sem,
				output_channels,
				previous_channels
		):

			corr = self.CatCorrAll(fc, sc, fc)
			decoder = OriginalVsReconstructed.UpBlock(fc + 1 + pc, oc, oc, b_upsample=(i != 1))

			cmis.append(corr)
			decs.append(decoder)


		self.cmis = nn.Sequential(*cmis)
		self.decs = nn.Sequential(*decs)
		self.final = nn.Conv2d(output_channels[-1], num_outputs, kernel_size=1)

	def forward(self, image, recon_image, labels, **_):

		if recon_image.shape != image.shape:
			recon_image = recon_image[:, :, :image.shape[2], :image.shape[3]]

		if not self.training:
			Pad = Padder(image.shape, 16)
			image, recon_image, labels = (Pad.pad(i.float()).type(i.dtype) for i in (image, recon_image, labels))

		#extracting features from original image, semantic labels and reconstructed image using VGG16 #521 step4
		vgg_img_feats = self.vgg_extractor(image)
		vgg_recon_feats = self.vgg_extractor(recon_image)
		sem_feats = self.sem_extractor(labels)

		value = None
		n_steps = self.cmis.__len__()

		for j in range(n_steps):
			i_inv = n_steps-(j+1)
			value = self.decs[j](self.cmis[j](value, vgg_img_feats[i_inv], vgg_recon_feats[i_inv], sem_feats[i_inv]))

		res = self.final(value)

		if not self.training:
			res = Pad.unpad(res)
		return res
