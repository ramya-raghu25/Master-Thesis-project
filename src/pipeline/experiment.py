"""
DESCRIPTION:     Python script for helper functions during training
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from .pipeline import *
from .config import *
from .transforms import *
import logging
log = logging.getLogger('exp')
from .logg import log_config_file
from pathlib import Path
from torch.optim import Adam as AdamOptimizer
import numpy as np
import torch
import os, gc, datetime, time, shutil
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GrumpyDict(dict):
	def __setitem__(self, key, value):
		if key not in self:
			log.warning('Dict Warning: setting key [{k}] which was not previously set'.format(k=key))
		super().__setitem__(key, value)

def train_state_initialize():
	train_state = GrumpyDict(
		epoch_idx = 0,
		best_loss_val = 1e5,
		run_name = 'training_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
	)
	return train_state

#Subclass for OriginalVsReconstructed and OriginalVsLabel and OriginalVsReconstructedAndLabel
class Experiment05():
	def __init__(self, cfg=None):
		cfg = cfg or self.cfg

		self.datasets = {}
		self.pipelines = {}
		self.initialize_config(cfg)
		self.initialize_transform()
	
	def initialize_config(self, cfg):
		self.cfg = cfg
		self.workdir = Path(cfg['dir_checkpoint'])
		self.workdir.mkdir(exist_ok=True, parents=True)
		(self.workdir / 'config.json').write_text(cfg_json_encode(self.cfg))

	#not needed
	def initialize_transform(self):
		pass


	def set_dataset(self, role, dset):
		self.datasets[role] = dset

	#function for loading checkpoint from respective directories
	def load_checkpoint(self, check_name ='check_best.pth'):
		dir_check = Path(self.cfg['dir_checkpoint'])
		path_check = dir_check / check_name
		torch.cuda.empty_cache()
		if path_check.is_file():
			log.info(f'Loading checkpoint found at {path_check}')
			return torch.load(path_check, map_location='cpu')

		else:
			log.info(f'No checkpoint found at {path_check}')
			return None

	#loads latest checkpoint to continue training
	def initialize_net(self, role):
		if role == 'train':
			check = self.load_checkpoint(check_name='check_last.pth')
			check_opt = self.load_checkpoint(check_name='optimizer.pth')
			
			self.build_network(role, check=check)
			self.build_optimizer(role, check_optimizer=check_opt)
			self.net_mod.train()

		elif role == 'eval':
			check = self.load_checkpoint(check_name='check_best.pth')
			self.build_network(role, check=check)
			self.net_mod.eval()

		else:
			raise NotImplementedError(f'role={role}')
		
		if check is not None:
			self.state = GrumpyDict(check['state'])
		else:
			self.state = train_state_initialize()

	# not needed
	def build_network(self, role, check=None, check_optimizer=None):
		log.info('Building network')


	@staticmethod
	#Loading checkpoints
	def load_checkpoint_to_net(net_mod, check_object):
		(missing_keys, superfluous_keys) = net_mod.load_state_dict(check_object['weights'], strict=False)
		if missing_keys:
			log.warning(f'Missing keys while loading a checkpoint: {missing_keys}')
		if superfluous_keys:
			log.warning(f'Missing keys while loading a checkpoint: {superfluous_keys}')

	#Building optimizer during training
	def build_optimizer(self, role, check_optimizer=None):
		log.info('Building optimizer')

		cfg_opt = self.cfg['train']['optimizer']

		network = self.net_mod
		self.optimizer = AdamOptimizer(
			[i for i in network.parameters() if i.requires_grad],
			lr=cfg_opt['learn_rate'],
			weight_decay=cfg_opt.get('weight_decay', 0),
		)
		self.learn_rate_scheduler = ReduceLROnPlateau(
			self.optimizer,
			patience=cfg_opt['lr_patience'],
			min_lr = cfg_opt['lr_min'],
		)

		if check_optimizer is not None:
			self.optimizer.load_state_dict(check_optimizer['optimizer'])


	def initialize_loss(self):
		log.info('Building loss_mod')

	def init_log(self, fids_to_display=[]):
		"""
		:param fids_to_display: ids of frames to view in tensorboard
		"""

		# log for the current training run
		self.tboard = SummaryWriter(self.workdir / f"tb_{self.state['run_name']}")
		# save ground truth to view in tensorboard
		self.tboard_gt = SummaryWriter(self.workdir / 'tb_gt')
		self.tboard_img = SummaryWriter(self.workdir / 'tb_img')
		self.train_out_dir = self.workdir / f"imgs_{self.state['run_name']}"
		self.train_out_dir.mkdir(exist_ok=True, parents=True)

		# names of the frames to display
		def short_frame_name(num):
			# remove directory path
			if '/' in num:
				num = os.path.basename(num)
			return num
			
		self.fids_to_display = set(fids_to_display)
		self.short_frame_names = {
			fid: short_frame_name(fid)
			for fid in self.fids_to_display
		}

	def log_selected_images(self, fid, frame, **_):
		if fid in self.fids_to_log:
			log.warning('log_selected_images: not implemented')

	def initialize_default_datasets(self):
		pass

	#training stage of uosn variants
	def initialize_pipelines(self):
		for role in ['train', 'val', 'test']:
			self.pipelines[role] = self.construct_uosn_pipeline(role)

	def get_epoch_limit(self):
		return self.cfg['train'].get('epoch_limit', None)

	#utilize CUDA
	def cuda_mod(self, attr_names):
		if torch.cuda.is_available():
			attr_names = [attr_names] if isinstance(attr_names, str) else attr_names

			for an in attr_names:
				setattr(self, an, getattr(self, an).cuda())

	
	def train_start_batch(self, **_):
		self.optimizer.zero_grad()
	
	def train_backpropagate(self, loss, **_):
		loss.backward()
		self.optimizer.step()

	def train_epoch_start(self, epoch_idx):
		self.net_mod.train()

	def train_epoch(self, epoch_idx):
		self.train_epoch_start(epoch_idx)

		out_frames = self.pipelines['train'].progress(
			dset = self.datasets['train'], 
			b_grad = True,
			b_pbar = False,
			b_accumulate = True,
			log_progress_interval = self.cfg['train'].get('progress_interval', None),
			short_epoch=self.cfg['train'].get('short_epoch_train', None),
		)
		gc.collect()

		results_avg = Frame({
			# avoid NAN in loss by taking average
			f: np.mean(np.array([pf[f] for pf in out_frames], dtype = np.float64))
			for f in out_frames[0].keys() if f.lower().startswith('loss')
		})

		self.train_epoch_finish(epoch_idx, results_avg)

		return results_avg['loss']

	def train_epoch_finish(self, epoch_idx, results_avg):
		for name, loss_avg in results_avg.items():
			self.tboard.add_scalar('train_'+name, loss_avg, epoch_idx)
	
	def val_epoch_start(self, epoch_idx):
		self.net_mod.eval()

	def val_epoch_finish(self, epoch_idx, results_avg):
		self.learn_rate_scheduler.step(results_avg['loss'])

		for name, loss_avg in results_avg.items():
			self.tboard.add_scalar('val_'+name, loss_avg, epoch_idx)

	def val_epoch(self, epoch_idx):
		self.val_epoch_start(epoch_idx)
		
		out_frames = self.pipelines['val'].progress(
			dset = self.datasets['val'], 
			b_grad = False,
			b_pbar = False,
			b_accumulate = True,
			short_epoch=self.cfg['train'].get('short_epoch_val', None),
		)
		gc.collect()

		results_avg = Frame({
			f: np.mean([pf[f] for pf in out_frames])
			for f in out_frames[0].keys() if f.lower().startswith('loss')
		})

		self.val_epoch_finish(epoch_idx, results_avg)

		return results_avg['loss']

	# training stage for uosn with swapped dataset (step 4)
	def run_epoch(self, epoch_idx):

		gc.collect()
		epoch_limit = self.get_epoch_limit()
		log.info('Epoch {ep:03d}{eplimit}\n	start training process....'.format(
			ep=epoch_idx,
			eplimit=f' / {epoch_limit}' if epoch_limit is not None else '',
		))
		t_train_start = time.time()
		loss_train = self.train_epoch(epoch_idx)
		gc.collect()
		t_val_start = time.time()
		log.info('	Training finished	t={tt}s	loss_t={ls}, starting validation'.format(
			tt=t_val_start - t_train_start,
			ls=loss_train,
		))

		gc.collect()
		loss_val = self.val_epoch(epoch_idx)
		gc.collect()
		log.info('	validation finished	t={tt}s	loss_e={ls}'.format(
			tt=time.time() - t_val_start,
			ls=loss_val,
		))

		is_best = loss_val < self.state['best_loss_val']
		if is_best:
			self.state['best_loss_val'] = loss_val
		is_check_scheduled = epoch_idx % self.cfg['train']['checkpoint_interval'] == 0

		if is_best or is_check_scheduled:
			self.save_checkpoint(epoch_idx, is_best, is_check_scheduled)

	def save_checkpoint(self, epoch_idx, is_best, is_scheduled):

		check_dict = dict()
		check_dict['weights'] = self.net_mod.state_dict()
		check_dict['state'] = dict(self.state)

		path_best = self.workdir / 'check_best.pth'
		path_last = self.workdir / 'check_last.pth'

		if is_scheduled:
			pytorch_save_atomic(check_dict, path_last)

			pytorch_save_atomic(dict(
					epoch_idx = epoch_idx,
					optimizer = self.optimizer.state_dict()
				), 
				self.workdir / 'optimizer.pth',
			)
		#update new checkpoint
		if is_best:
			log.info('	New best checkpoint generated')
			if is_scheduled:
				shutil.copy(path_last, path_best)
			else:
				pytorch_save_atomic(check_dict, path_best)

	#training the 3 variants of unknown object segmentation network
	def training_run(self, b_initial_eval=True):
		name = self.cfg['name']
		log.info(f'Experiment {name} - starting train')

		path_stop = self.workdir / 'stop'

		if b_initial_eval:
			loss_val = self.val_epoch(self.state['epoch_idx'])

			log.info(' loss_e={le}'.format(le=loss_val))
			self.state['best_loss_val'] = loss_val
		else:
			self.state['best_loss_val'] = 1e4

		b_continue = True

		while b_continue:			
			self.state['epoch_idx'] += 1
			self.run_epoch(self.state['epoch_idx'])

			if path_stop.is_file():
				log.info('Stop file detected')
				path_stop.unlink() # remove file
				b_continue = False

			epoch_limit = self.get_epoch_limit()
			if (epoch_limit is not None) and (self.state['epoch_idx'] >= epoch_limit):
				log.info(f'Reached Epoch limit {epoch_limit}')
				b_continue = False

	@classmethod
	# training procedure for 3 unknown object segmentation variants
	def training_procedure(cls):
		print(f'---- Start training for {cls.__name__} variant----')
		exp = cls()

		#loads corresponding experiments and configuration
		log_config_file(exp.workdir / 'training_file.log')

		try:
			#load train and validation images from cityscapes
			exp.initialize_default_datasets()

			#load network, optimizer and corresponding unknown object segmentation variant
			exp.initialize_net("train")
			log.info(f'Name of the experiment: {exp.state["run_name"]}')

			#create required output directory
			exp.initialize_transform()

			#load class weights from config file
			exp.initialize_loss()

			#log everything to view in tensorboard
			exp.init_log()

			# pipeline for unknown object segmentation variants
			exp.initialize_pipelines()

			#start training
			exp.training_run()

		# The following exception is raised if training crashes
		except Exception as e:
			log.exception(f'Exception occured during training: {e}')

	def load_args_for_role(self, role):
		if role == 'train':
			return  dict(
				shuffle = True,
				batch_size = self.cfg['net']['batch_train'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = True,
			)
		elif role == 'val' or role == 'test':
			return  dict(
				shuffle = False,
				batch_size = self.cfg['net']['batch_eval'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = False,
			)
		else:
			raise NotImplementedError("role: " + role)

	def construct_uosn_pipeline(self, role):
		# pipeline for training unknown object segmentation variants
		if role == 'train':
			tr_batch = Chain([
				SendCUDA(),
				self.train_start_batch,
				self.net_mod,
				self.loss_mod,
				self.train_backpropagate,
			])
			tr_output = Chain([
				KeepFieldsByPrefix('loss'),  # save loss for averaging later
				TrNP(), # clear away the gradients if any are left
			])
			
		elif role == 'val':
			tr_batch = Chain([
				SendCUDA(),
				self.net_mod,
				self.loss_mod,
			])
			tr_output = Chain([
				self.log_selected_images,
				KeepFieldsByPrefix('loss'), # save loss for averaging later
				TrNP(), # clear away the gradients if any are left
			])

		elif role == 'test':
			tr_batch = Chain([
				SendCUDA(),
				self.net_mod,
			])
			tr_output = Chain([
				TrNP(),
				untorch_images,
			])
	
		return Pipeline(
			tr_batch = tr_batch,
			tr_output = tr_output,
			loader_args = self.load_args_for_role(role),
		)


def pytorch_save_atomic(data, filepath):
	#Save previous checkpoint
	filepath = Path(filepath)
	filepath_temp = filepath.with_suffix('.tmp')
	torch.save(data, filepath_temp)
	shutil.move(filepath_temp, filepath)
