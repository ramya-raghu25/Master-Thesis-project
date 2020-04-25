"""
DESCRIPTION:     Python file for loading all pretrained models of Unknown Object Segmentation Network variants
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from ..pipeline import *
from ..pipeline.utils import bind
from ..pipeline.transforms import ChannelLoad, ChannelSave
from .networks import *
from ..paths import data_dir, cityscapes_dir
from ..datasets.cityscapes import DatasetCityscapesCompressed, CityscapesLabelInfo
from ..datasets.dataset import imwrite, ChannelLoaderImg, ChannelResultImg, imread, SemSegLabelTranslation
from ..sem_seg.networks import ClassifierSoftmax, CrossEntropy
from ..sem_seg.experiments import Colorimg
from ..reconstruction.experiments import Pix2PixHD_GAN, instances_from_semantics

from matplotlib import pyplot as plt
import numpy as np
from math import floor
from scipy import stats
from random import choice
import os, cv2

CMAP_MAGMA = plt.get_cmap('magma') 

channel_labels_swap = ChannelResultImg('swap/labels', suffix='_trainIds', img_ext='.png')
channel_recon_swap = ChannelResultImg('swap/recon', suffix='_recon')

#SubClass for OriginalVsReconstructed and OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment04(Experiment05):
	cfg = add_experiment(
		name='orig_vs_recon_model',
	)

	# 3 testing stage of all 3 variants
	def initialize_transform(self):
		super().initialize_transform()
		self.softmax_class = ClassifierSoftmax()
		self.cuda_mod(['softmax_class'])

		self.preprocess = Chain(
			label_to_validEval,
			get_errors,
			errors_to_gt,
		)

		self.postprocess_log = Chain(
			TrNP(),
		)

	def initialize_loss(self):
		print("Loading class weights")
		class_weights = self.cfg['train'].get('class_weights', None)
		if class_weights is not None:
			print('	Class weights are:', class_weights)
			class_weights = torch.Tensor(class_weights)
		else:
			print('	no class weights found')
		self.loss_mod = CrossEntropy(weight=class_weights)
		self.cuda_mod(['loss_mod'])

	#initialize network for correlating original image and reconstructed image during testing
	def net(self, image, recon_image, **_):
		return dict(
			pred_unknown_logits = self.net_mod(image, recon_image)
		)

	def loss(self, semantic_errors_label, pred_unknown_logits, **_):
		return self.loss_mod(pred_unknown_logits, semantic_errors_label)

	def classify(self, pred_unknown_logits, **_):
		# the unknown class prob is labelled as "1",
		return dict(
			unknown_p =self.softmax_class(pred_unknown_logits)['pred_prob'][:, 1, :, :]
		)

	# building net for original vs reconstructed uosn
	def build_network(self, role, check=None, check_optimizer=None):
		self.net_mod = OriginalVsReconstructed()
		if check is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(check['weights'])
		self.cuda_mod(['net_mod'])

	# training stage for 3 uosn variants with swapped dataset
	def init_log(self, log_frames=None):
		if log_frames is not None:
			self.log_frames = set(log_frames)

		super().init_log()

		data = self.datasets['val']

		channels_backup = data.channels_enabled
		data.set_enabled_channels('image', 'semantic_errors')

		# write the ground-truth also for comparison
		for fid in self.log_frames:
			remove_slash = str(fid).replace('/', '__')
			fr = data.get_frame_by_fid(fid)

			fr.apply(self.preprocess)
			imwrite(self.train_out_dir / f'gt_image_{remove_slash}.webp', fr.image)
			imwrite(self.train_out_dir / f'gt_labels_{remove_slash}.png', (fr.semantic_errors > 0).astype(np.uint8) * 255)
			self.tboard_img.add_image(	'{0}_img'.format(fid),	fr.image.transpose((2, 0, 1)),
				0,
			)

			self.tboard_gt.add_image('{0}_gt'.format(fid),	fr.semantic_errors[None, :, :],
				0,
			)

		data.set_enabled_channels(*channels_backup)

	# training stage for uosn  # step 3
	def eval_batch_log(self, frame, fid, unknown_p, **_):
		if fid in self.log_frames:
			frame.apply(self.postprocess_log)

			remove_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			# drop the alpha channel
			pred_colorimg = CMAP_MAGMA(frame.unknown_p, bytes=True)[:, :, :3]
			imwrite(self.train_out_dir / f'e{epoch:03d}_unknownP_{remove_slash}.webp', pred_colorimg)

			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.unknown_p[None, :, :],
				self.state['epoch_idx'],
			)


#Subclass for OriginalVsReconstructed and OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment03(Experiment04):
	cfg = add_experiment(
		name='orig_vs_recon_model',
		train=dict(
			class_weights=[1.45693524, 19.18586532],
			optimizer=dict(
				lr_patience=5,
			)
		)
	)

	def setup_dataset(self, dataset):
		print("setup_dataset")
		dataset.add_channels(
			pred_labels = channel_labels_swap,
			recon_image = channel_recon_swap,
		)
		dataset.post_load_pre_cache.append(
			SemSegLabelTranslation(fields=['pred_labels'], table=CityscapesLabelInfo.table_label_to_trainId),
		)
		dataset.discover()

	def initialize_default_datasets(self, b_threaded=False):
		print("initialize_default_datasets")
		# Cityscapes with prediction channel
		dset_train = DatasetCityscapesCompressed(split='train', b_cache=b_threaded)
		dset_val = DatasetCityscapesCompressed(split='val', b_cache=b_threaded)
		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dataset(dset)

		self.log_frames = set([dset_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)


#Subclass for OriginalVsReconstructed and OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment02(Experiment03):
	cfg = add_experiment(Experiment03.cfg,
						 name='orig_vs_recon_model',
						 train=dict(
			class_weights=[1.45693524, 19.18586532],
			optimizer=dict(
				lr_patience=5,
			)) )

	def setup_dataset(self, dataset): #also used in creating swapped data
		print("setup_dataset")
		dataset.add_channels(
			pred_labels=channel_labels_swap,
			recon_image=channel_recon_swap,
		)
		dataset.post_load_pre_cache.append(
			SemSegLabelTranslation(fields=['pred_labels'], table=CityscapesLabelInfo.tabulate_label_to_trainId),
		)
		dataset.discover()

#Subclass for OriginalVsReconstructed and OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment01(Experiment02):
	#This class creates the swapped training images for training the unknown object detector
	cfg = add_experiment(Experiment02.cfg,
						 name='orig_vs_recon_model',
						 gen_name = 'UOSN',
						 gen_img_ext='.webp',  #jpg
						 pix2pix_variant = 'pix2pixHD',
						 net=dict(
			batch_eval=3,
			batch_train=2,  # to train on small gpu
			num_classes=19, # num semantic classes
		),
						 disap_fraction = 0.5,
						 epoch_limit = 50,
						 )

	test_fields = ['image', 'recon_image']
	training_fields = ['image', 'recon_image', 'semantic_errors_label']

	# 2nd testing stage of all 3 variants
	# training stage for all 3 variants  #initializing transforms
	def initialize_transform(self):
		super().initialize_transform()
		self.init_uosn_dataset_channels()

		#training stage     #swap labels to create swapped_labels_dataset
		self.swap_mod = partial(disappear_objects, disappear_fraction=self.cfg['disap_fraction'])
		self.roi_out = np.logical_not(roi)
		self.preprocess = Chain()
		self.input_train = self.semantic_errors_to_label
		self.input_test = Chain()
		merge = Chain(
			ZeroCenterImgs(),
			torch_images,
		)

		self.pre_merge_testing = merge.copy()
		self.pre_merge_testing.append(
			KeepFields(*self.test_fields),
		)
		self.pre_merge_training = merge.copy()
		self.pre_merge_training += [
			KeepFields(*self.training_fields),
		]

	# set up data channels for 3 variants of unknown object segmentation network
	def init_uosn_dataset_channels(self):
		from pathlib import Path
		recon_name = self.cfg['gen_name']

		#Directory of the swapped_labels_dataset
		dir_swapped_dset = Path(data_dir, 'Swapped_Labels_Dataset', '{dset.name}', recon_name) #swapped_labels_dataset

		# Channels of the swapped_labels_dataset
		# labels with swapped instances
		self.ch_labelsSwap = ChannelLoaderImg(	dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_swapTrainIds.png')

        #label errors
		self.ch_uosn_mask = ChannelLoaderImg(dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_errors.png')

		# colored label errors
		self.ch_uosn_mask_color = ChannelLoaderImg(dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_errors_color.png')

		# reconstructed image from the swapped labels
		self.ch_recon = ChannelLoaderImg(
			dir_swapped_dset / 'recon_image' / '{dset.split}' / '{fid}_gen{channel.img_ext}',
			img_ext=self.cfg['gen_img_ext'])

		# colored labels with swapped instances
		self.ch_labelsSwap_colorimg = ChannelLoaderImg(dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_swapTrainIds_colorimg.png')

		# GT label maps
		# alternatively predicted label maps can also be used
		self.ch_labelsPred = ChannelLoaderImg(	dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_predTrainIds.png')

		# colored GT label maps
		self.ch_labelsPred_colorimg = ChannelLoaderImg(dir_swapped_dset / 'labels' / '{dset.split}' / '{fid}_predTrainIds_colorimg.png')

	# training stage for 3 variants of uosn with swapped dataset	#step 1
	def semantic_errors_to_label(self, semantic_errors, **_):
		errs = (semantic_errors > 0).astype(np.int64)
		errs[self.roi_out] = 255
		return dict(
			semantic_errors_label=errs,
		)

	#Setting up dataset for training uosn
	def setup_dataset(self, dataset):
		super().setup_dataset(dataset)
		dataset.add_channels(
			labels_swapErr_trainIds=self.ch_labelsSwap,
			semantic_errors=self.ch_uosn_mask,
			recon_image=self.ch_recon,
		)
		dataset.post_load_pre_cache = Chain()
		dataset.set_enabled_channels('image', 'recon_image', 'semantic_errors')
		dataset.discover()

	#Loading cityscapes for generating swapped training images
	def initialize_default_datasets(self, b_threaded=False):
		train_set = DatasetCityscapesCompressed(
			split='train',
			b_cache=b_threaded,
		)
		val_set = DatasetCityscapesCompressed(
			split='val',
			b_cache=b_threaded,
		)
		sets = [ train_set, val_set]#
		for i in sets:
			self.setup_dataset(i)

		self.log_frames = set([val_set.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', train_set)
		self.set_dataset('val', val_set)


	#initialize pipeline for training UOSN with swapped labels
	def uosn_dataset_init_pipeline(self, use_gt_labels=True, need_orig_label=True):
		"""
		:param use_gt_labels: True:  GT semantic labels of Cityscapes
		:param need_orig_label: True : Save all images in color
		"""
		self.pix2pix = Pix2PixHD_GAN(self.cfg['pix2pix_variant'])
		if use_gt_labels:
			self.load_correct_labels = Chain(

				# load cityscapes labels and instances
				ChannelLoad('labels_source', 'labels_source'),
				ChannelLoad('instances', 'instances'),

				# convert to trainIDs
				SemSegLabelTranslation(fields=dict(labels_source='pred_labels_ID'),
									   table=CityscapesLabelInfo.tabulate_label_to_trainId),
			)
		else:
			self.load_correct_labels = ChannelLoad(self.ch_labelsPred, 'pred_labels_ID'),

		self.alter_labels_and_recon_image = Chain(
			# load original labels
			self.load_correct_labels,
			# alter labels
			self.swap_mod,
			# reconstruct image
			bind(self.pix2pix.tr_gan_np, pred_labels_ID='labels_swapErr_trainIds').outs(
				recon_image='recon_image'),
		)

		self.swap_and_save = Chain(
			self.alter_labels_and_recon_image,

			ByField('semantic_errors', lambda x: (x > 0).astype(np.uint8) * 255),

			# saving swapped images
			ChannelSave(self.ch_uosn_mask, 'semantic_errors'),
			ChannelSave(self.ch_recon, 'recon_image'),
			ChannelSave(self.ch_labelsSwap, 'labels_swapErr_trainIds'),

			Colorimg('labels_swapErr_trainIds'),
			ChannelSave(self.ch_labelsSwap_colorimg, 'labels_swapErr_trainIds_colorimg')
		)

		# saving colored swapped images for visualization
		if need_orig_label:
			self.swap_and_save += [

				Colorimg('pred_labels_ID'),
				ChannelSave(self.ch_labelsPred, 'pred_labels_ID'),
				ChannelSave(self.ch_labelsPred_colorimg, 'pred_labels_trainIds_colorimg'),

				Colorimg('semantic_errors'),
			    ChannelSave(self.ch_uosn_mask_color, 'semantic_errors_colorimg'),
			]

	#step 1 loading dataset for training the unknown object segmentation network
	def generate_swapped_dataset(self, dsets=None, need_orig_label=True):
		self.uosn_dataset_init_pipeline(need_orig_label=need_orig_label)

		dsets = dsets or self.datasets.values()
		for dataset in dsets:
			dataset.set_enabled_channels()
			dataset.discover()
			Frame.frame_listapply(self.swap_and_save, dataset, n_proc=1, n_threads=1, ret_frames=False)

	def build_network(self, role, check=None, check_optimizer=None):
		#testing stage
		#performing correlation for original vs reconstructed image
		self.net_mod = OriginalVsReconstructed(num_outputs=2, freeze=True)
		if check is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(check['weights'])
		self.cuda_mod(['net_mod'])


	def initialize_loss(self):
		#training stage for all 3 variants of UOSN
		class_weights = self.cfg['train'].get('class_weights', None)
		if class_weights is not None:
			print('	Class weights are:', class_weights)
			class_weights = torch.Tensor(class_weights)
		else:
			print('	no class weights found!')

		self.loss_mod = CrossEntropy(weight=class_weights)
		self.cuda_mod(['loss_mod'])


	def net(self, image, recon_image, **_):
		#testing stage
		#performing correlation for original vs reconstructed image
		return dict(
			pred_unknown_logits=self.net_mod(image, recon_image)
		)

	# training stage for uosn with swapped dataset
	# step 2
	def loss(self, semantic_errors_label, pred_unknown_logits, **_):
		print("loss")
		return self.loss_mod(pred_unknown_logits, semantic_errors_label)

	def classify(self, pred_unknown_logits, **_):
		#Find softmax probs for all 3 variants during testing
		#print("#Find softmax probs for all 3 variants during testing")
		return dict(
			unknown_p =self.softmax_class(pred_unknown_logits)['pred_prob'][:, 1, :, :]
		)

	#constructing a pipeline for all 3 variants of unknown object semseg during testing
	def construct_uosn_pipeline(self, role):
		print("Constructing pipeline to detect unknown objects!")
		if role == 'test':
			return Pipeline(
				tr_input = self.input_test,
				batch_pre_merge= self.pre_merge_testing,
				tr_batch = Chain(
					SendCUDA(),
					self.net,
					self.classify,
					KeepFields('unknown_p'),
					TrNP(),
				),
				tr_output = Chain(

				),
				loader_args = self.load_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.input_train,
				batch_pre_merge= self.pre_merge_training,
				tr_batch = Chain(
					SendCUDA(),
					self.net,
					self.loss,
					self.classify,
					KeepFieldsByPrefix('loss', 'unknown_p'),
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
					self.input_train,
					RandomCrop(crop_size = self.cfg['train'].get('crop_size', [384, 768]), fields = self.training_fields),
					RandomlyFlipHorizontal(self.training_fields),
				),
				batch_pre_merge= self.pre_merge_training,
				tr_batch = Chain(
					SendCUDA(),
					self.train_start_batch,
					self.net,
					self.loss,
					self.train_backpropagate,
					KeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = Chain(
					KeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.load_args_for_role(role),
			)

#Subclass for OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment06(Experiment01):
	cfg = add_experiment(Experiment01.cfg,
						 name='orig_vs_label_model',
						 )

	test_fields = ['labels_swapErr_trainIds', 'image']
	training_fields = ['labels_swapErr_trainIds', 'image', 'semantic_errors_label']

	def initialize_transform(self):
		#testing stage of 2 variants : original vs label & original vs reconstructed and label
		super().initialize_transform()

	#setting up dataset for training uosn
	def setup_dataset(self, dataset):
		super().setup_dataset(dataset)
		dataset.channel_enable('labels_swapErr_trainIds')
		dataset.channel_disable('recon_image', 'pred_labels_ID')

	def build_network(self, role, check=None, check_optimizer=None):
		#testing stage
		#compares original vs label
		self.net_mod = OriginalVsLabel(
			num_outputs=2, freeze=True,
			num_classes=self.cfg['net']['num_classes'],
		)

		if check is not None:
			print('Loading pre trained weights from checkpoint')
			self.net_mod.load_state_dict(check['weights'])
		self.cuda_mod(['net_mod'])

	def net(self, labels_swapErr_trainIds, image, **_):
		# testing stage
		# performing correlation for original vs reconstructed image
		return dict(
			pred_unknown_logits = self.net_mod(labels_swapErr_trainIds, image)
		)

	def construct_uosn_pipeline(self, role):
		#testing stage
		# constructing a pipeline for 2 variants: original vs reconstructed & original vs reconstructed and label
		pipe = super().construct_uosn_pipeline(role)

		if role == 'test':
			pipe.batch_pre_merge.insert(0, RenameKw(pred_labels_ID ='labels_swapErr_trainIds'))
		return pipe


#Subclass for OriginalVsReconstructedAndLabel
class Experiment07(Experiment06):
	cfg = add_experiment(Experiment06.cfg,
						 name='orig_vs_recon_and_label_model',
						 )

	test_fields = ['labels_swapErr_trainIds', 'recon_image', 'image']
	training_fields = ['labels_swapErr_trainIds', 'recon_image', 'image', 'semantic_errors_label']

	def initialize_transform(self):
		# testing stage
		# initializing transforms for OriginalVsReconstructedAndLabel
		super().initialize_transform()

	#setting up dataset for training uosn
	def setup_dataset(self, dataset):
		super().setup_dataset(dataset)

		dataset.channel_enable('labels_swapErr_trainIds', 'recon_image')
		dataset.channel_disable('pred_labels_ID')

	def build_network(self, role, check=None, check_optimizer=None):
		#building net for OriginalVsReconstructedAndLabel
		self.net_mod = OriginalVsReconstructedAndLabel(
			num_outputs=2, freeze=True,
			num_classes=self.cfg['net']['num_classes'],
		)

		if check is not None:
			print('Loading pretrained weights from checkpoint')
			self.net_mod.load_state_dict(check['weights'])
		self.cuda_mod(['net_mod'])

	def net(self, labels_swapErr_trainIds, image, recon_image, **_):
		return dict(
			pred_unknown_logits = self.net_mod(image, recon_image, labels_swapErr_trainIds)
		)


#Main class for OriginalVsReconstructedAndLabel
class orig_vs_recon_and_label_model(Experiment07):
	cfg = add_experiment(Experiment07.cfg,
						 name='orig_vs_recon_and_label_model',
						 gen_name='UOSN',
						 swap_fraction=0.5,
						 )

	def initialize_transform(self):
		# testing stage
		# initializing transforms for original vs labels & reconstructed
		super().initialize_transform()
		self.swap_mod = partial(swap_labels_1, swap_fraction = self.cfg['swap_fraction'])

#Main class for OriginalVsLabel
class orig_vs_label_model(Experiment06):
	cfg = add_experiment(Experiment01.cfg,
						 name='orig_vs_label_model',
						 gen_name='UOSN',
						 )

	def initialize_transform(self):
		#testing stage
		#initializing transforms for original vs labels
		super().initialize_transform()
		self.swap_mod = None

#Main class for OriginalVsReconstructed
class orig_vs_recon_model(Experiment01):
	cfg = add_experiment(Experiment01.cfg,
						 name='orig_vs_recon_model',
						 gen_name='UOSN',
						 swap_fraction = 0.5,
						 )

	def initialize_transform(self):
		#testing stage of original vs reconstructed
		super().initialize_transform()
		self.swap_mod = partial(swap_labels_1, swap_fraction = self.cfg['swap_fraction'])


#########The following section contains functions for swapping labels and creating swapped labels dataset#######
def label_to_validEval(labels, dset, **_):
    #reshape label IDS
    v = dset.label_info.valid_in_eval_trainId[labels.reshape(-1)].reshape(labels.shape)
    return dict(
        labels_validEval=v,
    )


def get_errors(labels, pred_labels, labels_valid_Eval, **_):
    #get semseg errors to print error image
    errs = (pred_labels != labels) & labels_valid_Eval
    return dict(
        semantic_errors=errs,
    )

def errors_to_gt(semantic_errors, labels_valid_Eval, **_):
    errs = semantic_errors.astype(np.int64)
    errs[np.logical_not(labels_valid_Eval)] = 255
    return dict(
        semantic_errors_label=errs,
    )

try:
    """
	ROI is the Region of Interest associated with the Cityscapes vehicle. It excludes the ego vehicle from evaluation.
	Since Lost And Found does not provide a ROI, Cityscapes ROI is reused.
	"""
    roi_path = os.path.join(cityscapes_dir, 'roi.png')
    roi = imread(roi_path).astype(np.bool)
except Exception as e:
    print(f'Cityscapes ROI image is not present at {roi_path}): {e}')
    roi = np.ones((512, 1024), dtype=np.bool)

roi_neg = ~roi
remove_trainIDs = [CityscapesLabelInfo.name2trainId[n] for n in ['person', 'rider', 'car', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']]
MORPH_KERNEL = np.ones((11, 11), np.uint8)

#performing swapping of labels to create the "swapped labelled dataset" for training uosn
def swap_labels_1(pred_labels_ID, instances=None, swap_fraction=0.2, **_):
    """
    :param swap_fraction: probability of swapping the labels
    """
    labels = pred_labels_ID.copy()
    labels[roi_neg] = 255

    if instances is None:
        inst_gt_class = False
        instances = instances_from_semantics(labels, min_size=750, allowed_classes=remove_trainIDs)['instances']
    else:
        inst_gt_class = True  # ground truth instances

    labels_swapped = Swap_Labels_2(
        pred_labels_ID,
        instances,
        only_objects=inst_gt_class,
        fraction=swap_fraction,
    )['labels_swapErr']

    return dict(
        instances=instances,
        labels_swapErr_trainIds=labels_swapped,
        semantic_errors=(labels != labels_swapped) & roi,
    )

#performing swapping of labels to create the "swapped labelled dataset" for training uosn
def Swap_Labels_2(labels_source, instances, instance_ids=None, only_objects=False, fraction=0.2,
				  target_classes=np.arange(19), invalid_class=255, **_):
    if instance_ids is None:
        instance_uniq = np.unique(instances)
        if only_objects:
            instance_uniq_objects = instance_uniq[instance_uniq >= 24000]
        else:
            instance_uniq_objects = instance_uniq[instance_uniq >= 1]

        if instance_uniq_objects.__len__() == 0:
            return dict(
                labels_swapErr=labels_source.copy(),
            )

        instance_ids = np.random.choice(instance_uniq_objects, floor(instance_uniq_objects.__len__() * fraction), replace=False)

    labels = labels_source.copy()

    for i in instance_ids:
        instance_mask = instances == i
        instance_view = labels[instance_mask]
        obj_class = stats.mode(instance_view, axis=None).mode[0]

        if obj_class != invalid_class:
            tc = list(target_classes)
            try:
                tc.remove(obj_class)
            except ValueError:
                print(f'Instance class {obj_class} not found in set of classes {target_classes}')
            new_class = choice(tc)

            labels[instance_mask] = new_class

    result = dict(
        labels_swapErr=labels
    )

    return result

def disappear_instance(labels_source, instances, instance_ids=None, only_objects=True, swap_fraction=0.5, **_):
    print("Removing instances")
    if instance_ids is None:
        instance_uniq = np.unique(instances)

        if only_objects:
            instance_uniq_objects = instance_uniq[instance_uniq >= 24000]
        else:
            instance_uniq_objects = instance_uniq[instance_uniq >= 1]

        if instance_uniq_objects.__len__() == 0:
            return dict(
                labels_swapErr=labels_source.copy(),
            )

    instance_ids = np.random.choice(instance_uniq_objects, int(instance_uniq_objects.__len__() * swap_fraction), replace=False)
    disappear_mask = np.any([instances == i for i in instance_ids], axis=0)

    obj_classes = np.unique(labels_source[disappear_mask])

    forbidden_class = remove_trainIDs
    mask_forbidden_class = np.any([labels_source == j for j in forbidden_class], axis=0)

    mask_no_label = mask_forbidden_class | disappear_mask
    mask_label = np.logical_not(mask_no_label)

    closest_dst, closest_labels = cv2.distanceTransformWithLabels(
        mask_no_label.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    BG_indices = closest_labels[mask_label]
    BG_labels = labels_source[mask_label]

    label_translation = np.zeros(labels_source.shape, dtype=np.uint8).reshape(-1)
    label_translation[BG_indices] = BG_labels

    label_recon = labels_source.copy()
    label_recon[disappear_mask] = label_translation[closest_labels.reshape(labels_source.shape)[disappear_mask]]

    result = dict(
        labels_swapErr=label_recon,
    )

    return result


def disappear_objects(pred_labels_ID, instances=None, disappear_fraction=0.5, **_):
    print("Removing objects")
    labels = pred_labels_ID.copy()
    labels[roi_neg] = 255
    if instances is None:
        instances_gt_class = False
        instances = instances_from_semantics(labels, min_size=750, allowed_classes=remove_trainIDs)['instances']
    else:
        instances_gt_class = True  # ground truth instances

    labels_disap = disappear_instance(
        pred_labels_ID,
        instances,
        only_objects=instances_gt_class,
        fraction=disappear_fraction,
    )['labels_swapErr']

    return dict(
        instances=instances,
        labels_swapErr_trainIds=labels_disap,
        semantic_errors=(pred_labels_ID != labels_disap) & roi,
    )