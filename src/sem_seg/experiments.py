"""
DESCRIPTION:     Python script for loading pretrained PSPNet and BayesianSegNet and initializing pipelines
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

import logging
log = logging.getLogger('exp')
from .networks import *
from ..datasets.dataset import ImageBackgroundService
from ..datasets.cityscapes import DatasetCityscapesCompressed, CityscapesLabelInfo
#from ..datasets.bdd100k import DatasetBDD_Segmentation
from pjval_ml.OSR.GAN.src.sem_seg.pytorch_semantic_segmentation import models as ptseg_archs
from ..pipeline import *

class ConvertLabelsToColor(ByField):
	def __init__(self, fields=[('pred_labels', 'pred_labels_color')], colors_by_classid=CityscapesLabelInfo.colors_by_trainId):
		"""
		:param colors_by_classid:  [num_classes x 3] uint8
		"""
		super().__init__(fields=fields)
		self.set_class_colors(colors_by_classid)

	def set_class_colors(self, colors):
		self.colors_by_classid = colors
		# include 255 for "unlabeled" areas
		# if dtype is not specified, it will be float64 and tensorboard will display it incorrectly
		self.colors_by_classid_extension = np.zeros((256, 3), dtype=self.colors_by_classid.dtype)
		self.colors_by_classid_extension[:self.colors_by_classid.shape[0]] = self.colors_by_classid

	def set_override(self, class_id, color):
		self.colors_by_classid_extension[class_id] = color

	def forward(self, field_name, pred_labels):
		sh = pred_labels.shape
		return self.colors_by_classid_extension[pred_labels.reshape(-1)].reshape((sh[0], sh[1], 3))

# converting to color
class Colorimg(ConvertLabelsToColor):
	def __init__(self, *fields, table=CityscapesLabelInfo.colors_by_trainId):
		super().__init__(fields=[(f, f'{f}_color') for f in fields], colors_by_classid=table)


def batch_softmax(pred_logits, **_):
	return dict(
		pred_prob = torch.nn.functional.softmax(pred_logits),
	)

def batch_argmax(pred_logits, **_):
	return dict(
		pred_labels = pred_logits.argmax(dim=1, keepdim=False).byte(),
	)

#Train the semantic segmentation networks
class ExperimentSemSeg(Experiment05):

	def initialize_transform(self):
		super().initialize_transform()

		self.tr_input = Chain()

		self.colorimg = ConvertLabelsToColor(
			colors_by_classid=CityscapesLabelInfo.colors_by_trainId, 
		)

		self.postprocess_log = Chain(
			TrNP(),
			self.colorimg,
		)

		self.prepare_batch_test = Chain(
			ZeroCenterImgs(),
			torch_images,
			KeepFields('image')
		)

		self.prepare_batch_train = Chain(
			ZeroCenterImgs(),
			torch_images,
			KeepFields('image', 'labels'),
		)

		self.augmentation_crop_and_flip = Chain(
			RandomCrop(crop_size = self.cfg['train'].get('crop_size', [540, 960]), fields = ['image', 'labels']),
			RandomlyFlipHorizontal(['image', 'labels']),
		)


	def apply_net(self, **kwargs):
		return self.net_mod(**kwargs)

	def initialize_loss(self):
		self.loss_mod = CrossEntropy()
		self.cuda_mod(['loss_mod'])

	def init_log(self, log_frames=None):
		super().init_log()

		log_frames = log_frames or self.log_frames
		self.log_frames = set(log_frames)

		# Write the ground-truth for comparison
		for fid in self.log_frames:
			remove_slash = str(fid).replace('/', '__')
			fr = self.datasets['val'].original_dataset().get_frame_by_fid(fid)

			labels_colorimg = self.colorimg.forward('', fr.labels)

			ImageBackgroundService.imwrite(self.train_out_dir / f'gt_image_{remove_slash}.webp', fr.image)
			ImageBackgroundService.imwrite(self.train_out_dir / f'gt_labels_{remove_slash}.png', labels_colorimg)
			
			self.tboard_gt.add_image(
				'{0}_img'.format(remove_slash),
				fr.image.transpose((2, 0, 1)),
				0,
			)

			self.tboard_gt.add_image(
				'{0}_class'.format(remove_slash),
				labels_colorimg.transpose((2, 0, 1)),
				0,
			)

	def eval_batch_log(self, frame, fid, **_):  #pred_prob
		if fid in self.log_frames:
			frame.apply(self.postprocess_log)

			remove_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			ImageBackgroundService.imwrite(self.train_out_dir / f'e{epoch:03d}_labels_{remove_slash}.png', frame.pred_labels_colorimg)

			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.pred_labels_colorimg.transpose((2, 0, 1)),
				epoch,
			)

	def construct_uosn_pipeline(self, role):

		if role == 'test':
			return Pipeline(
				tr_input = self.tr_input,
				batch_pre_merge= self.prepare_batch_test,
				tr_batch = Chain(
					SendCUDA(),
					self.apply_net,
					batch_softmax,
					batch_argmax,
					KeepFields('pred_prob', 'pred_labels'),
					TrNP(),
				),
				tr_output = Chain(
					Colorimg('pred_labels'),
				),
				loader_args = self.load_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_input,
				batch_pre_merge= self.prepare_batch_train,
				tr_batch = Chain(
					AsType({'labels': torch.LongTensor}),
					SendCUDA(),
					self.apply_net,
					self.loss_mod,
					batch_argmax,
					KeepFieldsByPrefix('loss', 'pred_labels'),
				),
				tr_output = Chain(
					self.eval_batch_log,
					KeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.load_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = Chain(
					self.tr_input,
					self.augmentation_crop_and_flip,
				),
				batch_pre_merge= self.prepare_batch_train,
				tr_batch = Chain(
					AsType({'labels': torch.LongTensor}), # long tensor error
					SendCUDA(),
					self.train_start_batch,
					self.apply_net,
					self.loss_mod,
					self.train_backpropagate,
					KeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = Chain(
					KeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.load_args_for_role(role),
			)

################################# PSPNET #################################
class SemSegPSP(ExperimentSemSeg):
	# change config for PSPNet here and config.py
	cfg = add_experiment(CONFIG_PSP,
		name='psp1',
		net = dict (
			batch_train = 3,
			batch_eval = 1,
		),
		train = dict (
			crop_size = [384, 768],
			epoch_limit = 50,
		),
	)

	def initialize_loss(self):
		self.loss_mod = LossPSP()
		self.cuda_mod(['loss_mod'])

	def build_network(self, role, check=None, check_optimizer=None):
		#Build net and optimizer while training
		self.net_mod = ptseg_archs.PSPNet(
			num_classes = self.cfg['net']['num_classes'], 
			pretrained = self.cfg['net'].get('backbone_pretrained', True) and role != 'eval', 
			use_aux = self.cfg['net'].get('use_aux', True), 
		)
		#not implemented
		if self.cfg['net'].get('backbone_freeze', False):
			log.info('Freezing backbone')
			for i in range(5):
				backbone_mod = getattr(self.net_mod, f'layer{i}')
				for param in backbone_mod.parameters():
					param.requires_grad = False

		if check is not None:
			log.info('Loading weights from checkpoint')
			#this value is none
			self.load_checkpoint_to_net(self.net_mod, check)
	
		self.cuda_mod(['net_mod'])

	def apply_net(self, image, **_):
		return dict(
			pred_logits = self.net_mod(image),
		)

	def setup_dset(self, dset):
		dset.discover()
		dset.load_class_statistics()

	def initialize_default_datasets(self, b_threaded=False):
		train_set = DatasetCityscapesCompressed(split='train', b_cache=b_threaded)
		val_set = DatasetCityscapesCompressed(split='val', b_cache=b_threaded)

		datasets = [train_set, val_set]
		for d in datasets:
			self.setup_dset(d)

		self.log_frames = set([val_set.frames[j].fid for j in [2, 3, 4, 5, 6, 8, 9]])
		self.set_dataset('train', train_set)
		self.set_dataset('val', val_set)


class ExpSemSegPSP_Ensemble(SemSegPSP):
	cfg = add_experiment(SemSegPSP.cfg,
		name='psp_cityscapes',
	)

	uncertainty_type = 'pred_variance_ensemble'

	@classmethod
	#create different ensemble models
	def create_sub_exp(cls, exp_id):
		cfg = add_experiment(cls.cfg,
			name = "{orig_name}_{i:02d}".format(
				orig_name=cls.cfg['name'],
				i=exp_id,
			),
		)
		return cls(cfg)

	#load all sub experiments for ensemble
	def load_sub_experiments(self, additional_subexp_names=[]):
		self.sub_exps = []
		name = self.cfg['name']

		work_dir = Path(self.workdir).parent
		sub_exp_dir = list(work_dir.glob(self.cfg['name'] + '_*'))
		sub_exp_dir += [work_dir / a for a in additional_subexp_names]

		for i in sub_exp_dir:
			cfg = json.loads((i / 'config.json').read_text())
			sub = self.__class__(cfg)
			self.sub_exps.append(sub)

		log.info('Found sub experiments for ensemble : {ses}'.format(ses=', '.join(se.cfg['name'] for se in self.sub_exps)))

	def initialize_net(self, role):
		if role == 'master_eval':
			for exp in self.sub_exps:
				exp.initialize_net('eval')
		else:
			super().initialize_net(role)

	def ensemble(self, image, **_):
		res = []
		for sub_exp in self.sub_exps:
			res.append(
				torch.nn.functional.softmax(
					sub_exp.apply_net(image=image)['pred_logits'],
					dim=1,
				),
			)

		res = torch.stack(res)
		avg = torch.mean(res, 0)
		var = torch.sum(torch.var(res, 0), 1)
		pred_entropy = -torch.sum(torch.mean(res, 0) * torch.log(torch.mean(res, 0)), dim=1)

		return dict(
			pred_prob = avg,
			pred_labels = avg.argmax(dim=1).byte(),
			pred_variance_ensemble = var,
			pred_entropy= pred_entropy,
				)

	def construct_uosn_pipeline(self, role):
		if role == 'test':
			return Pipeline(
				tr_input=Chain(
				),
				batch_pre_merge=Chain(
					ZeroCenterImgs(),
					torch_images,
					KeepFields('image')
				),
				tr_batch=Chain(
					SendCUDA(),
					self.ensemble,
					KeepFields('pred_labels', 'pred_variance_ensemble', 'pred_entropy'),
					TrNP(),
				),
				tr_output=Chain(
					Colorimg('pred_labels')
				),
				loader_args=self.load_args_for_role(role),
			)
		else:
			return super().construct_uosn_pipeline(role)


class SemSegPSPEnsembles(ExpSemSegPSP_Ensemble):
	cfg = add_experiment(SemSegPSP.cfg,
		name='pspnet_model',
		epoch_limit=20,
	)

################################# BayeisanSegNet #################################
class ExpSemSegBayes(ExperimentSemSeg):
	cfg = add_experiment(CONFIG_PSP,
		name='BayesSemSeg',
		net = dict(
			batch_eval = 1,
			batch_train = 4,
		),
		train = dict (
			crop_size = [384, 768],
		),
	)
	torch.cuda.empty_cache()
	uncertainty_type = 'pred_variance_dropout'

	def setup_dset(self, dataset):
		dataset.discover()

	def build_network(self, role, check=None, check_optimizer=None):
		#Build net and optimizer for training """
		torch.cuda.empty_cache()
		self.net_mod = BayesianSegNet(self.cfg['net']['num_classes'], pretrained=True)
		torch.cuda.empty_cache()
		if check is not None:
			log.info('Loading weights from checkpoint')
			self.net_mod.load_state_dict(check['weights'])
		torch.cuda.empty_cache()
		self.cuda_mod(['net_mod'])

	def net(self, image, **_):
		result = self.net_mod(image)
		return dict(
			pred_logits = result,
		)

	def net_with_uncertainty(self, image, **_):
		result = self.net_mod.forward_multisample(image)

		return dict(
			pred_prob = result['mean'],
			pred_labels = result['mean'].argmax(dim=1).byte(),
			pred_variance_dropout = torch.sum(result['var'], 1),
		    pred_entropy = -torch.sum(result['mean'] * torch.log(result['mean']), dim=1),
		)

	def construct_uosn_pipeline(self, role):
		if role == 'test':
			return Pipeline(
				tr_input = Chain(),
				batch_pre_merge= self.prepare_batch_test,
				tr_batch = Chain(SendCUDA(),
								 self.net_with_uncertainty,
								 KeepFields('pred_labels', 'pred_variance_dropout', 'pred_entropy'),
								 TrNP(),
								 ),
				tr_output = Chain(
					Colorimg('pred_labels'),
				),
				loader_args = self.load_args_for_role(role),
			)
		else:
			return super().construct_uosn_pipeline(role)


class SemSegBayseg(ExpSemSegBayes):
	cfg = add_experiment(ExpSemSegBayes.cfg,name='bayesiansegnet_model',
	)