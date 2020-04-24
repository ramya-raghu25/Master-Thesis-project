"""
DESCRIPTION:     Python script for logging functions
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

import logging, sys

def log_config_default(log_name):
	# write INFO and below to stdout
	handler_stdout = logging.StreamHandler(sys.stdout)
	handler_stdout.setFormatter(
		logging.Formatter(fmt='{message}', style='{'),
	)
	handler_stdout.addFilter(lambda r: r.levelno <= logging.INFO)
	
	# write WARNING and above to stderr
	handler_stderr = logging.StreamHandler(sys.stderr)
	handler_stderr.setFormatter(
		logging.Formatter(fmt='{levelname:<7}|{filename}:{lineno}| {message}', style='{'),
	)
	handler_stderr.setLevel(logging.WARNING)
	
	# Root log is not used here since all libraries are writing trash to it
	# Use the 'exp' for "experiment" log space
	log = logging.getLogger(log_name)
	log.setLevel(logging.DEBUG)
	for hs in [handler_stdout, handler_stderr]: log.addHandler(hs)
	return log

log = log_config_default('exp')

def log_config_file(filepath, log_obj = log):
	handler_file = logging.FileHandler(filepath)
	handler_file.setFormatter(logging.Formatter(
		fmt = '{asctime}|{levelname:<7}| {message}',
		style = '{',
		datefmt = '%m-%d %H:%M:%S'
	))
	
	# add handlers to root
	log_obj.addHandler(handler_file)
	log_obj.info(f'Log file {filepath} has been initialized')
