"""
DESCRIPTION:     Python file for calculating metrics for Unknown Object Segmentation Network
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

from ..pipeline.transforms import Base, Show
from ..pipeline.utils import adapt_to_img_data
import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2 as cv
from pathlib import Path

levels = 1024

#extract features from confusion matrix
def binary_confusion_mat(prob, gt_label, roi=None, levels=levels):
	if roi is not None:
		prob = prob[roi]
		area = prob.__len__()
		gt_label = gt_label[roi]
	else:
		area = np.prod(prob.shape)

	gt_bool_label = gt_label.astype(np.bool)

	gt_true_area = np.count_nonzero(gt_label)
	gt_false_area = area - gt_true_area

	prob_true = prob[gt_bool_label]
	prob_false = prob[~gt_bool_label]

	tp, _ = np.histogram(prob_true, levels, range=[0, 1])
	tp = np.cumsum(tp[::-1])

	fn = gt_true_area - tp

	fp, _ = np.histogram(prob_false, levels, range=[0, 1])
	fp = np.cumsum(fp[::-1])

	tn = gt_false_area - fp

	conf_mat = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1)

	conf_mat = conf_mat.astype(np.float64) / area
	return dict(
		conf_mat=conf_mat,
	)

#calculate AUROC
def calculate_auroc(roc_info, label, name, plot=None, fmt=None):
	fp_rates, tp_rates, num_levels = (roc_info[met] for met in ['fp_rates', 'tp_rates', 'num_levels'])
	#fp_rates, tp_rates, num_levels, recall, precision = (roc_info[k] for k in ['fp_rates', 'tp_rates', 'num_levels', 'recall', 'precision'])
	#recall, precision, num_levels = (roc_info[k] for k in ['recall', 'precision', 'num_levels'])

	if plot is None:
		fig, plot = plt.subplots(1)
		plot.set_xlim([0, 1])
		plot.set_ylim([0, 1])

	fmt_args = []
	fmt_kwargs = {}

	if fmt is not None:
		if isinstance(fmt, str):
			# one string specifier
			fmt_args = [fmt]
		elif isinstance(fmt, dict):
			fmt_kwargs = fmt
		elif isinstance(fmt, tuple):
			fmt_kwargs = dict(color=fmt[0], linestyle=fmt[1])
		else:
			raise NotImplementedError(f"Must format object {fmt}")

	if name=='aoc':
		#Trapezoidal rule to find AUC is implemented here
		area_under_curve = np.trapz(tp_rates, fp_rates)
		#area_under_curve = np.trapz(precision, recall)

		plot.plot(fp_rates, tp_rates,
			*fmt_args,
			label='{lab}  {a:.02f}'.format(lab=label, a=area_under_curve),
			**fmt_kwargs,

		)
		plot.set_xlim([0, 1])
		plot.set_ylim([0, 1])
	# else:
	# 	area_under_curve = np.trapz(precision, recall)
	# 	plot.plot(recall, precision,
	# 			  *fmt_args,
	# 			  label='{lab}  {a:.02f}'.format(lab=label, a=area_under_curve),
	# 			  **fmt_kwargs,
	# 			  )
	# 	xlimit = max(recall[-2] + 0.05, plot.get_xlim()[1])
	#
	# 	plot.set_xlim([0, xlimit])

	return area_under_curve

#calculate evaluation metrics
def conf_mat_to_roc(name, conf_mats):
	# roi_area = np.count_nonzero(roi) if roi is not None else 1
	num_levels = conf_mats.shape[0]
	tp = conf_mats[:, 0, 0]
	fp = conf_mats[:, 0, 1]
	fn = conf_mats[:, 1, 0]
	tn = conf_mats[:, 1, 1]

	precision = tp/(tp+fp)#best is 1
	recall = tp / (tp + fn)  #best is 1
	specificity = tn / (tn + fp)  #best is 1
	#print("Precision :",np.mean(precision),"Recall/Sensitivity:",np.mean(recall),"Specificity/TNR:",np.mean(specificity))

	accuracy = (tp + tn) / (tp +tn + fp +fn)#best is 1
	# err_rate = (fp + fn) / (tp +tn + fp +fn)  #best is 0
	print("Accuracy:", np.mean(accuracy))

	tp_rates = tp / (tp+fn)#best is 1
	fp_rates = fp / (fp+tn)#best is 1
	#print(name,"tp_rates",np.mean(tp_rates),"fp_rates/1-tnr",np.mean(fp_rates))

	F1_measure = (2*(np.mean(precision)) *(np.mean(recall)))/(np.mean(recall)+np.mean(precision))
	IOU = tp / (tp+fp+fn)
	print("Mean IOU:",np.mean(IOU))
	print("F1_measure:",F1_measure)

	return dict(
		name = name,
		num_levels=num_levels,
		tp_rates=tp_rates,
		fp_rates=fp_rates,
		conf_mats=conf_mats,
		#
		precision=precision,
		recall=recall,
		accuracy=accuracy,
		specificity = specificity,
		F1_measure=F1_measure,
		IOU = IOU
	)

#############The following functions are used for plotting the metrics##############

def ensure_numpy_image(img):
	if isinstance(img, torch.Tensor):
		img = img.cpu().numpy().transpose([1, 2, 0])
	return img

# create image grid
def img_grid(imgs, num_cols=2, downsample=1):
	num_images = imgs.__len__()
	num_rows = int(np.ceil(num_images / num_cols))
	image_size = np.array(imgs[0].shape[:2]) // downsample
	full_size = (num_rows * image_size[0], num_cols * image_size[1], 3)
	out = np.zeros(full_size, dtype=np.uint8)
	pos_row_col = np.array([0, 0])
	for image in imgs:
		# none for black section
		if image is not None:
			image = ensure_numpy_image(image)
			if downsample != 1:
				image = image[::downsample, ::downsample]
			image = adapt_to_img_data(image)

			tl = image_size * pos_row_col
			br = tl + image_size

			out[tl[0]:br[0], tl[1]:br[1]] = image

		pos_row_col[1] += 1
		if pos_row_col[1] >= num_cols:
			pos_row_col[0] += 1
			pos_row_col[1] = 0
	return out


class ImageGrid(Show):
	def __init__(self, channel_names, out_name='demo', num_cols=2, downsample=1):
		super().__init__(*channel_names)
		self.out_name = out_name
		self.num_cols = num_cols
		self.downsample = downsample

	def __call__(self, **fields):
		imgs = self.get_channel_values(fields, self.channel_names)
		grid = img_grid(imgs, num_cols=self.num_cols, downsample=self.downsample)

		return {
			self.out_name: grid,
		}


class Blend(Base):
	def __init__(self, field_1, field_2, field_out, alpha_a=0.8):
		self.field_1 = field_1
		self.field_2 = field_2
		self.field_out = field_out
		self.alpha_a = alpha_a

	def __call__(self, **fields):
		img_1 = adapt_to_img_data(fields[self.field_1])
		img_2 = adapt_to_img_data(fields[self.field_2])
		blend_img = cv.addWeighted(img_1, self.alpha_a, img_2, 1 - self.alpha_a, 0.0)

		return {
			self.field_out: blend_img,
		}


# plot and save the ROCs
def draw_roc_curve(infos, save=None, figsize=(500, 500)):
	if isinstance(infos, dict):  # this is false
		infos = list(infos.values())

	dpi = 96
	fig1, plot = plt.subplots(1, figsize=tuple(s / dpi for s in figsize), dpi=dpi)
	areas = []
	# pr = []
	for info in infos:
		label = info.get('plot_label', info['name'])
		fmt = info.get('plot_fmt', None)
		aoc = calculate_auroc(info, label=label, plot=plot, fmt=fmt, name='aoc')
		# prcurve = roc_plot_additive(info, label=label, plot=plot, fmt=fmt, name ='pr' )
		areas.append(aoc)
	# pr.append(prcurve)

	b = [0.50]
	areas.append(b)
	a = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	plot.plot(a, a, label='Random classifer 0.50')  # plot baseline classifier

	plot.set_xlabel('False Positive Rate')
	plot.set_ylabel('True Positive Rate')

	# Sorting variants based on highest AUC
	permutation = np.argsort(areas)[::-1]
	# permutation = np.argsort(pr)[::-1]

	handles, labels = plot.get_legend_handles_labels()
	handles = [handles[k] for k in permutation]
	labels = [labels[k] for k in permutation]
	#print("handles",handles)
	#print("labels",labels)
	legend = plot.legend(handles, labels, loc=(0.5, 0))
	fig1.tight_layout()

	if save:
		save = Path(save)
		fig1.savefig(save.with_suffix('.png'))
	return fig1