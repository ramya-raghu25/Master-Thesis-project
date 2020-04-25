# Master-Thesis-Project
# Open Set Recognition in Semantic Segmentation

  Deep Neural Networks have recently shown great success in many machine learning domains and problems. Traditional machine learning models
assume that the classes encountered during testing is always a subset of the
classes encountered during training, or in other words, a closed set. However,
this assumption is violated often as real world tasks in computer vision often come across the Open Set Recognition (OSR) problem where incomplete
knowledge of the world and several unknown inputs exists, i.e. never-beforeseen classes are introduced during testing. This requires the model to accurately classify the known classes and effectively deal with the unknown classes.

  Applying these models for safety-critical applications, such as autonomous driving and medical analysis, pose many safety challenges due to prediction uncertainty and misclassification errors. These erroneous information, can lead to poor decisions, causing accidents and casualties. Hence, the ability to localize the unknown objects for such mission critical applications would be highly beneficial. This thesis investigates a methodology that tackles detecting unknown objects for a semantic segmentation task.

  It builds on the intuition that, during segmentation, a network ideally
produces erroneous labels or misclassifications in regions where the unknown
objects are present. Therefore, reconstructing an image from the resulting
predicted semantic label map (which has misclassified the unknown objects),
should produce significant discrepancies with respect to the input image.
Hence, the problem of segmenting these unknown objects is reformulated as
localizing the poorly reconstructed regions in an image. The effectiveness
of the proposed method is demonstrated using two segmentation networks
evaluated on two publicly available segmentation datasets. Using the Area
Under the Receiver Operating Characteristic (AUROC) metric, it is proved
that this method can perform significantly better than standard uncertainty
estimation methods and the baseline method.


# Motivation

  Artificial Intelligence (AI) systems are now being given control in safetycritical scenarios such as autonomous driving and medical diagnosis where errors in the output
of an algorithm could cause accidents or even casualties. In fact, detection
failure in perception systems is the main reason behind recent reported selfdriving accidents, and addressing this issue turns
out to be extremely challenging. The inherent risk when using AI systems in
such areas is that neural networks are not robust or fault tolerant. That is,
they output a confidence value that cannot always be trusted. These systems
themselves cannot detect new data and would only identify it based on what
they are trained on.

  In the case of autonomous driving, the models depend on low level feature extraction methods like image segmentation to process their raw inputs.
Based on the outputs of these models, high level decisions are carried out
by the car. An algorithm able to detect an object out of the ordinary, will
enable the autonomous systems to prevent accidents or other critical events
either by alerting its driver or taking other fail-safe maneuvers. If the model
responsible for the low level tasks fails to detect these “unknown” objects
and produces erroneous predictions, this can lead to catastrophic events. An
example of such an event was when a Tesla autonomous car was involved in a
fatal crash because the system did not detect a white truck against a cloudy
bright sky [Vlasic and Boudette, 2016](https://www.nytimes.com/2016/07/01/business/self-driving-tesla-fatal-crash-investigation.html). If the system could have identified
it as an “unknown” or incorporated a high level of uncertainty (or low confidence) in its prediction, the disaster might have been avoided.

  The root cause for such incidents is due to the fact that they occur under
open set conditions, i.e. an environment that the network has never seen
before. When a network encounters an “unknown” object during test time,
it will be forced to misclassify it as one among the trained known classes.
This is called as the Open Set Recognition (OSR) problem. This thesis will
try to implement a novel method in order to tackle the OSR problem by
detecting or localizing the “unknown” rather than misclassifying it into one
of the “known” classes, for such safety critical applications.


# Unknown Object Segmentation Network

  This thesis aims to efficiently identify unknown objects during test time in
a semantic segmentation task and to predict the probability that a certain
pixel belongs to an unknown class. This is in contrast to most of the segmentation tasks, which focus on assigning a probability to each pixel, of it
belonging to a class it has seen in training, without any provision for the
unknown.

  The original image is shown to a semantic segmentation network to obtain a semantic label map. Then this map is passed to a generative network that attempts to reconstruct the original image. The idea here is that if the image contains objects belonging to a class that the segmentation network has not been trained on, the corresponding pixels should be misclassified in the semantic label map and therefore be poorly reconstructed. The next ideal step would be to efficiently identify these unknown objects by detecting significant pixel differences between the original image and the reconstructed image. Here, a novel network called the UOSN is introduced, to detect these differences and estimate their uncertainties.
<img src="Results/UOSN architecture.PNG" width="100%" />

## Directory structure

`Master-Thesis-Project`
* `src`
  * `sem_seg` - Semantic segmentation using PSPNet and Bayesian SegNet
  * `reconstruction` - Image to image reconstruction from predicted labels
  * `uosn` - Unknown Object Segmentation Network
  * `paths.py` - Give respective directories
  * `datasets` - Scripts for loading all datasets
    * `Lost and Found dataset` - [Lost and Found](http://www.6d-vision.com/lostandfounddataset)
    * `Cityscapes dataset` - [Cityscapes dataset](https://www.cityscapes-dataset.com/)
    * `MSCOCO dataset` - [MSCOCO](http://cocodataset.org/#home)
    * `NYU Depth dataset` - [NYU Depth Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
 

## Dataset pre-processing

The datasets were used in 1024x512 resolution, while the original formats are 2048x1024. The conversion scripts are available in [Master-Thesis-project/training_scripts](Master-Thesis-project/training_scripts).
The script needs a [webp encoder](https://anaconda.org/conda-forge/libwebp) and [imagemagick](https://imagemagick.org/index.php).


## Generate swapped labelled dataset

To generate the swapped labelled dataset for training the 3 variants of UOSN, run [Master-Thesis-project/training_scripts/gen_swapped_data.py](Master-Thesis-project/training_scripts/gen_swapped_data.py)


## Training the networks

* To train the Semantic segmentation networks (PSPNet and Bayesian SegNet), run [Master-Thesis-project/training_scripts/semseg_train.py](Master-Thesis-project/training_scripts/semseg_train.py). The training scripts are taken from (https://github.com/zijundeng/pytorch-semantic-segmentation).

* To train the Pix2PixHD GAN, run [Master-Thesis-project/training_scripts/pix2pixHD_train.sh](Master-Thesis-project/training_scripts/pix2pixHD_train.sh). The training scripts are taken from (https://github.com/NVIDIA/pix2pixHD).

* To train all the 3 variants of UOSN using the swapped labelled dataset, run [Master-Thesis-project/training_scripts/train_uosn.py](Master-Thesis-project/training_scripts/train_uosn.py)

* Weights will be written to `exp_dir`. Checkpoints are saved every epoch as follows:
  * `chk_best.pth` - Checkpoint with the lowest loss on eval set
  * `chk_last.pth` - Checkpoint after the most recent epoch
  * `optimizer.pth` - Optimizer data (momentum etc.) after the most recent epoch
  * `training.log` - Stores logs from the logging module. If the training procedure failed, the exception can be viewed here.

*The loss is written to tensorboard and can be displayed in the following way:

```bash
	tensorboard --logdir $exp_dir/name_of_exp
```


## Running the experiment

Simply run the script in [Master-Thesis-project/training_scripts/main.py](Master-Thesis-project/training_scripts/main.py). Specify the dataset to be used in dset (LAF or NYU) and the semantic segmentation network to be used in semseg_variants (PSPNet or BaysegNet).


## Results

This folder contains the output images highlighting the unknown objectsfrom the experiments performed on Lost and Found and NYU dataset along with some failure cases. It also contains the AUROC performance curves for UOSN, UOSN architecture and few examples from the swapped labelled dataset.