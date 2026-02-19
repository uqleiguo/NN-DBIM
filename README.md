# **NN-DBIM**
This repository includes the codes that implement the two papers, "Medical Microwave Imaging Using Physics-Guided Deep Learning Part 1: The Forward Solver" and "Medical Microwave Imaging Using Physics-Guided Deep Learning Part 2: The Inverse Solver" by [Lei Guo](https://about.uq.edu.au/experts/15936), [Alina Bialkowski](https://about.uq.edu.au/experts/19747), and [Amin Abbosh](https://about.uq.edu.au/experts/1576). You can find our two papers using the following links.
- [Medical Microwave Imaging Using Physics-Guided Deep Learning Part 1: The Forward Solver](https://ieeexplore.ieee.org/abstract/document/11316536)
- [Medical Microwave Imaging Using Physics-Guided Deep Learning Part 2: The Inverse Solver](https://ieeexplore.ieee.org/abstract/document/11352785)
## **Environment**
The python codes are based on torch 2.8.0, numpy 2.0.2, matplotlib 3.9.4, and python 3.9. The matlab codes are based on MATLAB_R2024a. No additional MATLAB toolbox is required.
## **Codes**
- 'NN_DBIM_forward_solver_model.py' and 'NN_DBIM_inverse_solver_model.py' are the neural network models of the forward and inverse solvers used in the distorted Born iterative method (DBIM).
- 'NN_DBIM_forward_solver_train.py' and 'NN_DBIM_inverse_solver_train.py' are the codes for training the forward and inverse neural network models.
- 'NN_DBIM_forward_solver_test.py' and 'NN_DBIM_inverse_solver_test.py' are the codes for testing the trained forward and inverse neural network models. Trained models are given [here](https://drive.google.com/drive/u/1/folders/1JLMeRyqpyMm8ai7RFCTH_uSmxt5xTN14) (Access will be given by contacting l.guo3@uq.edu.au)
- 'FDTD_solver_interface.m' and 'func_FDTD.m' are the codes for the conventional FDTD forward solver.
- 'MoM_solver_interface.m' and 'func_MoM.m' are the codes for the conventional MoM forward solver.
