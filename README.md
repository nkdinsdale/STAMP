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
@article{DINSDALE2022102583,
title = {STAMP: Simultaneous Training and Model Pruning for low data regimes in medical image segmentation},
journal = {Medical Image Analysis},
pages = {102583},
year = {2022},
doi = {https://doi.org/10.1016/j.media.2022.102583},
author = {Nicola K. Dinsdale and Mark Jenkinson and Ana I.L. Namburete}
}


