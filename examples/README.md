Teo Price-Broncucia

# Overview of Examples

This `examples` directory contains cleaned up code regarding the usage of adaptive ODE solvers in machine learning. The scripts in this directory assume that `torchdiffeq` is installed following instructions from the main directory.

## Demo
Two additional examples have been added. ode_demo_1pend.py and latent_ode_2pend.py
To visualize the training progress, run
```
python ode_demo_1pend.py --viz
```
Demos will store training loss as pickled file. With visualization option images
of training will be stored. If desired one can use helper script img_to_video.py
to create movie showing training progression.
