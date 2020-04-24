"""
DESCRIPTION:     Python script for calculating loss for segmentation architectures
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

This code is taken and modified from https://github.com/zijundeng/pytorch-semantic-segmentation which is licensed
under the MIT License.

Date: 20.10.2019

Copyright (c) 2017 ZijunDeng

For details on the license please have a look at MasterThesis/Licenses/MIT_License.txt
"""

import torch
from torch import nn
from pjval_ml.OSR.GAN.src.sem_seg.pytorch_semantic_segmentation import models as ptseg_models
from ..pipeline.utils import Padder

class CrossEntropy(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_softmax = nn.LogSoftmax(dim=1) # calculate along channels
		self.nll_loss = nn.NLLLoss(
			weight,
			reduction='mean' if size_average else 'none',
			ignore_index=ignore_index,
		)

	def forward(self, pred_logits, labels, **other):
		return dict(loss = self.nll_loss(
			self.log_softmax(pred_logits),
			labels,
		))


class ClassifierSoftmax(nn.Module):
	def __init__(self):
		super().__init__()
		self.softmax = torch.nn.Softmax2d()

	def forward(self, pred_logits, **_):
		if pred_logits.shape.__len__() == 4:
			pred_softmax = self.softmax(pred_logits)
		else:
			pred_softmax = self.softmax(pred_logits[None])[0]  # calculate for a single sample, with no batch dimension

		return dict(
			pred_prob = pred_softmax
		)

class LossPSP(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_softmax = nn.LogSoftmax(dim=1) # calculate along channels
		self.nll_loss = nn.NLLLoss(weight, reduction='mean' if size_average else 'none', ignore_index=ignore_index)
		self.cel = CrossEntropy(weight, size_average, ignore_index)

	def forward(self, pred_logits, labels, **other):
		if isinstance(pred_logits, tuple):
			pred_main_raw, pred_aux_raw = pred_logits

			loss_main = self.nll_loss(self.log_softmax(pred_main_raw), labels)
			loss_aux = self.nll_loss(self.log_softmax(pred_aux_raw), labels)

			return dict(
				loss = loss_main * (1.0/1.4) + loss_aux * (0.4/1.4),
				loss_main = loss_main,
				loss_aux = loss_aux,
			)
		else:
			return self.cel(pred_logits, labels, **other)


class BayesianSegNet(ptseg_models.SegNetBayes):
	def forward(self, img):  #not used #testing

		padder = Padder(img.shape, 32)
		img = padder.pad(img)

		result = super().forward(img)
		return padder.unpad(result)
