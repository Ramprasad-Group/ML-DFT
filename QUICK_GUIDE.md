# QUICK GUIDE 

If you just downloaded and installed the ML-DFT package and do not care about how it works but just want to start making electronic structure predictions, this is your guide. For more details about the package, and to use it without employing your own resources, follow the tutorials.

## Making predictions with trained models from paper
You will need the following files in the folder you are going to make the predictions on:

* predict.csv : contains the direction to the POSCAR files with the structures
* KS_emulator.py : code to run. No need to modify anything
* inp_params.py : Set of input commands to specify the code what to do. Look below to see what each command means.

## Training (and making predictions)
You will need the following files in the folder you are going to retrain the models (and predict):

* Train.csv and Val.csv: location of POSCAR files to use for training and validation.
* (predict.csv : contain the direction to the POSCAR files with the structures)
* KS_emulator.py : code to run. No need to modify anything
* inp_params.py : Set of input commands for the code. Look below to see what each command means.

IMPORTANT!: Depending on what model you want to retrain you will need the following files in your database:
* Training potential energy model: You will need to have an 'energy' file with the reference potential energy, a 'forces' file with the reference atomic forces, and a 'stress' file with the stress tensor components (XX,YY,ZZ,XY,YZ,ZX).
* Training DOS and VB/CB models: You will need to have a 'dos' file containing the DOS values every 0.1 eV, after being shifted by the energy in vacuum. Also a 'VB_CB' file containing the valence band energy and conduction band energy. 

For more info on the format of the necessary files for training, you can look at examples in the tutorials/database folder.

## List of Input commands in inp_params.py

*train_e*: do you want to retrain the potential energy, forces and stress model? True or False
*ert_epochs*: number of epochs when retraining the potential energy model
*ert_batch_size*: batch size for retraining the potential energy model
*ert_patience*: how many epochs to wait without imrpoving the validation loss until the potential energy stops retraining
*train_dos*: do you want to retrain the DOS and VBM/CBM model? Is either True or False 
*drt_epochs*: number of epochs when retraining the DOS model
*drt_batch_size*: batch size for retrainign the DOS model
*drt_patience*: how many epochs to wait without imrpoving the validation loss until the DOS stops retraining
*new_weights_e*: do you have new weights for the potential energy from a previous retraining you would like to use for testing? True or False
*new_weights_dos*: do you have new weights for the DOS and VB/CB from a previous retraining you would like to use for testing? True or False
*test_chg*: do you want to predict the valence electron density? True or False
*test_e*: do you want to predict the potential energy, atomic forces and stress tensor? True or False
*test_dos*: do you want to predict the DOS, VBM and CBM? True or False

*cut_off_rad*: distance at which we stop considering neighboring atoms during fingerprinting
*batch_size_fp*: largest amount of atoms of each element in the structure (for fingerprinting)
*widest_gaussian*: size of the widest gaussian when fingerprinting 
*narrowest_gaussian*: size of narrowest gaussian when fingerprinting
*num_gamma*: number of gamma values to consider when fingerprinting

*plot_dos*: do you want to plot the DOS? True or False
*comp_chg*: do you want to compare the predicted valence electron density with a reference charge densiy? True or False
*write_chg*: do you want to write the valence electron density into a set of grid points in 3D? True or False
*ref_chg*: do you want to write the predicted valence electron density onto the same grid as a reference electron density? True or False.
*grid_spacing*: if ref_chg=False, you can choose the grid spacing for the grid points in 3D. We recommend between 0.2 and 0.7

## Running ML_DFT

Once you have the necessary files and command values in inp_params.py, just do:

```angular2
python ML_DFT.py
```  

If you run into an issue or have any questions about the code, we first recommend going through the tutorials. If it does not solve your problem, please contact the authors.
