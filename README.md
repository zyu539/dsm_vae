## Description

This is the git repository for the DSM-VAE model. The model means to
extend the existing Deep Survival Machine model (https://arxiv.org/abs/2003.01176) by adding a VAE structure to it.

To be noticed that the model is build upon the DSM model and tested on SUPPORT dataset. The auton-survival package is
required to run the model. The auto-survival package is already included in this repository, which
is slightly modified to be compatible to the new  VAE loss. The original package can
be found at https://github.com/autonlab/auton-survival