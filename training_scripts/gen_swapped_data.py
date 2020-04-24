"""
DESCRIPTION:     Python file for generating swapped training images
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""


from src.uosn.experiments import orig_vs_recon_and_label_model
from src.pipeline.config import add_experiment

if __name__ == '__main__':

    class Swapped_Labels_Dataset(orig_vs_recon_and_label_model):
        cfg = add_experiment(orig_vs_recon_and_label_model.cfg,
            name = 'Swapped_Labels_Dataset',
            gen_name = 'UOSN',
            gen_img_ext = '.webp', # better compression
            swap_fraction = 0.75, # 0.5
        )

    eval = Swapped_Labels_Dataset()

    # Load the Cityscapes dataset
    eval.initialize_default_datasets()

    #Generate and store the swapped labels for training uosn
    eval.generate_swapped_dataset(dsets=eval.datasets.values())

    print("Dataset created!!")

