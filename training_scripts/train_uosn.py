"""
DESCRIPTION:     Python file for training all 3 variants of Unknown Object Segmentation Network
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.
"""

def original_reconstruct():
    if __name__ == '__main__':
        from src.uosn import orig_vs_recon_model as model
        model.training_procedure()

def original_label():
    if __name__ == '__main__':
        from src.uosn import orig_vs_label_model as model
        model.training_procedure()

def original_reconstruct_label():
    if __name__ == '__main__':
        from src.uosn import orig_vs_recon_and_label_model as model
        model.training_procedure()


#uncomment the network you want to train

original_reconstruct()
#original_label()
#original_reconstruct_label()

