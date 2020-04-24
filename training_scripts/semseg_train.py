"""
DESCRIPTION:     Python file for training PSPNet and BayesianSegNet
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

"""

from src.paths import *
from src.pipeline import *
from src.sem_seg import *
from src.pipeline.config import add_experiment


def train_PSP(ensemble_id=0):
	b_threaded = False

	model = SemSegPSPEnsembles

	print(ensemble_id)
	cfg = add_experiment(model.cfg,
	    name = "{norig}_{i:02d}".format(norig=model.cfg['name'], i=ensemble_id),
	)
	eval = model(cfg)

	# load train and validation images from the dataset
	eval.initialize_default_datasets(b_threaded)

	# load network and optimizer
	eval.initialize_net("train")

	# create required output directory
	eval.initialize_transform()

	# load class weights from config file
	eval.initialize_loss()

	# log everything to view in tensorboard
	eval.init_log()

	# pipeline for training
	eval.initialize_pipelines()

	# start training
	eval.training_run()

def train_BSN():
	b_threaded = False

	eval = SemSegBayseg()

	# load train and validation images from the dataset
	eval.initialize_default_datasets(b_threaded)

	# load network and optimizer
	eval.initialize_net("train")

	# create required output directory
	eval.initialize_transform()

	# load class weights from config file
	eval.initialize_loss()

	# log everything to view in tensorboard
	eval.init_log()

	# pipeline for training
	eval.initialize_pipelines()

	if b_threaded:
		eval.pipelines['train'].loader_class = SamplerThreaded
		eval.pipelines['val'].loader_class = SamplerThreaded

	# start training
	eval.training_run()

#Training function for PSPNet. Specify the model number here "ensemble_id"
train_PSP(ensemble_id=0)

#Training function for BayesianSegNet.
train_BSN()
