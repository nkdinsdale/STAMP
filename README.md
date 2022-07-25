# STAMP: Simultaneous Training and Model Pruning
## Code for implementation of Simultaneous Model Pruning and Training for Low Data Regimes in Medical Image Segmentation

This is a working release. Any issues please contact: nicola.dinsdale@cs.ox.ac.uk. Further code will be added in time. 

![GitHub Logo](graphical_abs.png)

Files
-----------------
- trainprune_main --> runs training procedure
  required arugments: 
  
    -m = pruning mode
    
    -r = number of recovery epochs
    
    -i = starting epoch
    
    -no = number of iterations to run
   
- pruning_functions.py --> functions controlling training

- pruning tools --> completes the filter pruning for unet arch

- model architecture --> unet arch adapted for targeted dropout
  
 
Software Versions 
-----------------
Python 3.5.2

PyTorch 1.0.1.post2

If you use code from this repository please cite:

```
@article{STAMP,
title = {STAMP: Simultaneous Training and Model Pruning for Low Data Regimes in Medical Image Segmentation},
journal = {BioArxiv},
year = {2021},
doi = {https://www.biorxiv.org/content/10.1101/2021.11.26.470124v2},
author = {Nicola K. Dinsdale and Mark Jenkinson and Ana I.L. Namburete}
}

