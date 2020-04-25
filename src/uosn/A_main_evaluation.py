"""
DESCRIPTION:     Python file for main evaluation for Unknown Object Segmentation Network
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

import numpy as np
from pathlib import Path
import logging, gc
log = logging.getLogger('exp.eval')
import os
from ..paths import exp_dir
from ..pipeline.config import add_experiment
from ..pipeline.frame import Frame
from ..pipeline.transforms import ChannelLoad, ChannelSave,RenameKw, Chain, KeepFields,KeepFieldsByPrefix
from ..pipeline.utils import bind
from ..datasets.dataset import  ChannelLoaderImg, ChannelLoadHDF5, hdf5_read, hdf5_write
from .metrics import binary_confusion_mat, conf_mat_to_roc, ImageGrid, Blend,draw_roc_curve# draw_unknown_contour
from ..sem_seg.experiments import SemSegPSPEnsembles, SemSegBayseg, SemSegPSP, ConvertLabelsToColor
from ..reconstruction.experiments import Pix2PixHD_GAN
from .experiments import orig_vs_recon_model, orig_vs_label_model, orig_vs_recon_and_label_model
from collections import namedtuple

#from ..baseline.experiments import baseline_OSW_model
#from ..datasets.dataset import SemSegLabelTranslation

class UnknownObjectSegmentation:
    #uncertainty metrics
    SEM_SEG_UNCERTAINTY_VARIANTS = {
        'BaySegnet': ['Uncertainty_Drop_Var','Uncertainty_Drop_Ent'],
        'PSPnet': ['Uncertainty_Ens_Var','Uncertainty_Ens_Ent'],
    }
    #unknown object detector variants and baseline
    UNKNOWN_VARIANTS = {
        'Orig_vs_Recon': orig_vs_recon_model,
        'Orig_vs_label': orig_vs_label_model,
        'Orig_vs_Recon_and_label': orig_vs_recon_and_label_model,
        #'Baseline_OSW': baseline_OSW_model,

    }
    #define plot styles for all methods
    Style = namedtuple('Style', ['display_name', 'display_fmt'])
    PLOT_STYLES = {
        'Orig_vs_Recon_and_label': Style('Orig_vs_Recon_and_label', dict(color='b', linestyle='-')),
        'Orig_vs_Recon': Style('Orig_vs_Recon', dict(color=(0.8, 0.3, 0.), linestyle='-.')),
        'Orig_vs_label': Style('Orig_vs_label', dict(color='g', linestyle='--')),
        #'Baseline_OSW': Style('Baseline_OSW', dict(color='r', linestyle='--')),
        'Uncertainty_Drop_Var': Style('Uncertainty_Drop_Var', dict(color='k', linestyle=':')),
        'Uncertainty_Ens_Var': Style('Uncertainty_Ens_Var', dict(color='k', linestyle=':')),
        'Uncertainty_Drop_Ent': Style('Uncertainty_Drop_Ent', dict(color='r', linestyle=':')),
        'Uncertainty_Ens_Ent': Style('Uncertainty_Ens_Ent', dict(color='r', linestyle=':')),
        'Random classifier': Style('Random classifier', dict(color='r', linestyle=':')),
    }
    def __init__(self, semseg_variants):
        self.workdir = os.path.join(exp_dir, 'Eval_Results') #Final results are stored here
        self.semseg_variants = semseg_variants
        self.uncertainty_variants = self.SEM_SEG_UNCERTAINTY_VARIANTS[self.semseg_variants]
        self.unknown_variants = (list(self.UNKNOWN_VARIANTS.keys()) + self.uncertainty_variants)
        #self.unknown_variants.sort()
        print("Variants used : ",self.unknown_variants)
        self.initialize_persistence()

    def initialize_persistence(self):

        #Storage locations for results and intermediate data
        out_default = os.path.join(exp_dir, 'Eval_Results', '{dset.name}_{dset.split}', f'Semseg_{self.semseg_variants}')
        out_dir = Path(out_default)
        self.base_dir = out_dir

        # output steps:
        self.storage = dict(
            # semantic segmentation image output
            pred_labels_ID = ChannelLoaderImg(out_dir / 'Semseg_labels/{fid}_PredLabels.png'),
            pred_labels_color = ChannelLoaderImg(out_dir / 'Semseg_labels/{fid}_PredColor.png'),

            # reconstructed image output
            recon_image = ChannelLoaderImg(out_dir / 'Reconstructed_images/{fid}_recon_image.webp'),

            #unknown object and uncertainty heatmap output
            Result1 = ChannelLoaderImg(out_dir / 'Results/{remove_slash}_Result1.webp'),
            Result2 = ChannelLoaderImg(out_dir / 'Results/{remove_slash}_Result2.webp'),
        )
        # unknown object scores
        self.storage.update({
            f'unknown_{name}': ChannelLoadHDF5(
                out_dir / f'unknown_score/unknown_{name}.hdf5',
                '{fid}',
                write_as_type=np.float16,
                read_as_type=np.float32,
            )
            for name in self.unknown_variants
        })
        for c in self.storage.values(): c.ctx = self

    # initialize semantic segmentation
    def initialize_semseg(self):
        #load checkpoints for BaySegnet
        if self.semseg_variants == 'BaySegnet':
            exp = SemSegBayseg()
            exp.initialize_net('eval')
            #BaySegnet uses dropout
            uncertainty_type = 'unknown_Uncertainty_Drop_Var'
            uncertainty_type1 = 'unknown_Uncertainty_Drop_Ent'
            renames = RenameKw(
                pred_labels = 'pred_labels_ID',
                pred_variance_dropout=uncertainty_type,
                pred_entropy=uncertainty_type1,
            )

        #load checkpoints for PSPnet
        elif self.semseg_variants == 'PSPnet':
            exp = SemSegPSPEnsembles()
            exp.load_sub_experiments()
            exp.initialize_net('master_eval')

            #PSPNet uses ensemble
            uncertainty_type = 'unknown_Uncertainty_Ens_Var'
            uncertainty_type1 = 'unknown_Uncertainty_Ens_Ent'
            renames = RenameKw(
                pred_labels = 'pred_labels_ID',
                pred_variance_ensemble = uncertainty_type,
                pred_entropy = uncertainty_type1,
            )
        else:
            raise NotImplementedError(self.semseg_variants)

        write_results = Chain(
            ChannelSave(self.storage['pred_labels_ID'], 'pred_labels_ID'),
            ChannelSave(self.storage['pred_labels_color'], 'pred_labels_color'),
            ChannelSave(self.storage[uncertainty_type], uncertainty_type),  #variance
            ChannelSave(self.storage[uncertainty_type1], uncertainty_type1),  #entropy

        )

        exp.out_for_eval = Chain(
            renames,
            write_results
        )

        self.exp_semseg = exp

    # perform semantic segmentation
    def eval_semseg(self, dset):
        pipe = self.exp_semseg.construct_uosn_pipeline('test')
        tr_out = self.exp_semseg.out_for_eval
        pipe.tr_output += tr_out
        dset.set_enabled_channels('image')
        pipe.progress(dset, b_accumulate=False)
        dset.out_hdf5_files()

    # initialize pix2pixHD
    def initialize_gan(self):
        self.pix2pix = Pix2PixHD_GAN()

    # perform image reconstruction
    def eval_gan(self, dset):
        pipe = self.pix2pix.construct_pix2pixHD_pipeline()

        pipe.tr_input += [ ChannelLoad(self.storage['pred_labels_ID'], 'pred_labels_ID'),
        ]
        pipe.tr_output.append( ChannelSave(self.storage['recon_image'], 'recon_image'),
        )

        pipe.progress(dset, b_accumulate=False)
        dset.out_hdf5_files()

    # initialize unknown object segmentation network
    def initialize_uosn(self, detector_name):
        # initialize unknown object segmentation network
        exp_class = self.UNKNOWN_VARIANTS[detector_name]
        exp = exp_class()
        exp.initialize_net('eval')
        score_field = f'unknown_{detector_name}'
        exp.out_for_eval = ChannelSave(self.storage[score_field], 'unknown_p')
        return exp

    # perform unknown object segmentation (single variant)
    def eval_uosn(self, exp_name, dset):
        if isinstance(exp_name, str):
            exp = self.initialize_uosn(exp_name)
        else:
            exp = exp_name
        pipe = exp.construct_uosn_pipeline('test')

        pipe.tr_input += [
            ChannelLoad(self.storage['pred_labels_ID'], 'pred_labels_ID'),
            ChannelLoad(self.storage['recon_image'], 'recon_image'),
        ]

        pipe.tr_output.append(exp.out_for_eval)
        pipe.progress(dset, b_accumulate=False)
        dset.out_hdf5_files()

    # perform unknown object segmentation (3 variants)
    def eval_uosn_all(self, dset):
        names = list(self.UNKNOWN_VARIANTS.keys())

        for name in names:
            log.info(f'Running {name}')
            score_field = f'unknown_{name}'
            score_file = Path(self.storage[score_field].resolve_file_path(dset, dset.frames[0]))
            if score_file.is_file():
                log.info(f'Output file for {name} already exists')
            else:
                self.eval_uosn(name, dset)
                gc.collect()

    # creating final heatmaps
    def store_images(self, dset):
        create_images = Chain(
            ChannelLoad('image', 'image'),
            ChannelLoad('labels_source', 'labels_source'),
            dset.get_unknown_gt,

            ChannelLoad(self.storage['pred_labels_ID'], 'pred_labels_ID'),
            ChannelLoad(self.storage['recon_image'], 'recon_image'),
            ChannelLoad(self.storage['unknown_Orig_vs_Recon_and_label'], 'unknown_Orig_vs_Recon_and_label'),
            ChannelLoad(self.storage['unknown_Orig_vs_label'], 'unknown_Orig_vs_label'),
            ChannelLoad(self.storage['unknown_Orig_vs_Recon'], 'unknown_Orig_vs_Recon'),

            #Must hard code this line!!!!!!! : unknown_Uncertainty_Drop_Var or unknown_Uncertainty_Ens_Var
            #ChannelLoad(self.storage[f'unknown_Uncertainty_Drop_Var'], 'unknown_Uncertainty_Drop_Var'),
            ChannelLoad(self.storage[f'unknown_Uncertainty_Ens_Var'], 'unknown_Uncertainty_Ens_Var'),

            # Must hard code this line!!!!!!! : unknown_Uncertainty_Drop_Ent or unknown_Uncertainty_Ens_Ent
            #ChannelLoad(self.storage[f'unknown_Uncertainty_Drop_Ent'], 'unknown_Uncertainty_Drop_Ent'),
            ChannelLoad(self.storage[f'unknown_Uncertainty_Ens_Ent'], 'unknown_Uncertainty_Ens_Ent'),

            #TrChannelLoad(self.storage['unknown_Baseline_OSW'], 'unknown_Baseline_OSW'),  #rbm
            #draw_unknown_contour,
            ConvertLabelsToColor([('pred_labels_ID', 'pred_labels_color')]),
        )

        create_images += [
            Blend(unknown_field, 'image', f'overlay_{unknown_field}', 0.8)
            for unknown_field in

            # Must hard code this line!!!!!!! : unknown_Uncertainty_Drop_Var or unknown_Uncertainty_Ens_Var
             #                                : unknown_Uncertainty_Drop_Ent or unknown_Uncertainty_Ens_Ent
            ['unknown_Orig_vs_Recon_and_label', 'unknown_Uncertainty_Ens_Var','unknown_Uncertainty_Ens_Ent', 'unknown_Orig_vs_label',
                'unknown_Orig_vs_Recon'] # 'unknown_Baseline_OSW'
        ]
        create_images += [
            ImageGrid([
                    'image', 'pred_labels_color',
                    'recon_image', 'overlay_unknown_Orig_vs_Recon_and_label',
                ],
                num_cols = 2,
                out_name = 'Result1',
            ),

            ImageGrid([
                # Must hard code this line!!!!!!! : overlay_unknown_Uncertainty_Drop_Var or overlay_unknown_Uncertainty_Ens_Var
                #                                : overlay_unknown_Uncertainty_Drop_Ent or overlay_unknown_Uncertainty_Ens_Ent
                'overlay_unknown_Orig_vs_Recon', 'overlay_unknown_Uncertainty_Ens_Var',
               'overlay_unknown_Orig_vs_label', 'overlay_unknown_Uncertainty_Ens_Ent',
                ],
                num_cols = 2,
                out_name = 'Result2',
            ),
        ]
        print("Saving output images")
        create_images += [
            ChannelSave(self.storage['Result1'], 'Result1'),
            ChannelSave(self.storage['Result2'], 'Result2'),
            # demo images are cpu-bound operations, hence perform multiprocessing
            # the dataset object contains HDF5 handles, which cannot be sent between processes
            KeepFields(),
        ]
        Frame.frame_listapply(create_images, dset.frames, n_proc=1, batch=4) #n_proc=8

    #calculate AUROC curves for all variants
    def calc_roc_curves_for_variants(self, dset, Unknown_Variants=None, on_road=False):
        if Unknown_Variants is None:
            Unknown_Variants = self.unknown_variants

        #roi_field = 'roi' is restricting evaluations only to the road
        roi_field = 'roi' if not on_road else 'roi_onroad'
        # Initialize Pipeline
        # Load groundtruths only once for parallel computation
        RoC = Chain(
            ChannelLoad('labels_source', 'labels_source'),
            dset.get_unknown_gt,
            dset.get_roi_frame,
        )

        # calculate confusion matrices for each variant
        for variant in Unknown_Variants:
            score_field = f'unknown_{variant}'
            conf_mat_field = f'conf_mats_{variant}'
            RoC += [
                ChannelLoad(self.storage[score_field], score_field),
                bind(binary_confusion_mat, prob=score_field, gt_label='unknown_gt', roi=roi_field).outs(conf_mat=conf_mat_field),
            ]

        # return only the confusion matrices (by clearing all the rest) to save memory
        RoC += [
            KeepFieldsByPrefix('conf_mats_'),
        ]

        # execute pipeline
        dset.discover()
        dset.out_hdf5_files()
        dset.set_enabled_channels()
        results = Frame.frame_listapply(RoC, dset.frames, ret_frames=True, n_proc=1, batch=8)

        # extract info from confusion matrices : tp,fp,tn,fn
        rocinfo_dict = {}
        for variant in Unknown_Variants:
            variant_name_out = variant if not on_road else f'{variant}_onroad'
            conf_mat_field = f'conf_mats_{variant}'
            conf_mat_sum = np.sum([fr[conf_mat_field] for fr in results], axis=0)

            rocinfo_dict[variant_name_out] = conf_mat_to_roc(
                name=variant_name_out,
                conf_mats=conf_mat_sum,
            )

        # store all ROC results
        self.roc_save_all(dset=dset, rocinfo_dict=rocinfo_dict)
        return rocinfo_dict

    #the following fuctions are for creating paths, loading, saving and plotting the AUROC curves
    def roc_path(self, variant_name, dset):
        path_temp = str(self.base_dir / 'unknown_roc' / f'{variant_name}_roc.hdf5')
        rocpath = Path(path_temp.format(dset=dset, channel=Frame(ctx=self)))
        return rocpath


    def roc_save(self, variant_name, dset, rocinfo):
        p = self.roc_path(variant_name, dset)
        log.info(f'Saving hdf5 under {p}')
        p.parent.mkdir(parents=True, exist_ok=True)
        hdf5_write(p, rocinfo)

    def roc_save_all(self, dset, rocinfo_dict):
        for name, info in rocinfo_dict.items():
            self.roc_save(variant_name=name, dset=dset, rocinfo=info)

    def roc_load(self, variant_name, dset):
        return hdf5_read(self.roc_path(variant_name, dset))

    def plot_path(self, dset):
        path_temp = str(self.base_dir / 'ROC_curve_{dset.name}_{dset.split}')
        path = Path(path_temp.format(dset=dset, channel=Frame(ctx=self)))
        return path

    def plot_roc(self, dataset, rocinfo_dict=None, variant_names=None):
        if rocinfo_dict is None:
            if variant_names is None:
                variant_names = self.unknown_variants
            #load all ROC information here
            rocinfo_dict = {
                name: self.roc_load(variant_name=name, dset=dataset)
                for name in variant_names
            }

        # load default styles and names for ROC plot
        infos_list = []
        for info in rocinfo_dict.values():
            style = self.PLOT_STYLES.get(info['name'], {})
            info_style = style._asdict()
            info_style.update(info)
            infos_list.append(info_style)
        print("Plotting ROC curves")
        roc = draw_roc_curve(infos_list, save=self.plot_path(dataset))

class PSP_Ensemble(SemSegPSP):
    #Load saved weights for PSP from the ensemble
    cfg = add_experiment(SemSegPSPEnsembles.cfg,
        name = 'pspnet_model_1',
        net = dict(
            use_aux = True,
            apex_mode = False,
        )
    )
    def training_run(self):
        raise NotImplementedError()