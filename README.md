# ML-DFT

ML-DFT is a combination of various deep learning models which predict various properties of the electronic structure of molecules and polymers at the DFT level: the electron density, density of states, and total potential energy (with forces and stress tensor). The only required input is the structure information in a POSCAR format. Additionally, the package allows the user to retrain some models using his/her own database. 


ML-DFT can be used to predict the electronic structure from classical MD simulations of structures too large for DFT. Additionally, it can be used to perform structure search where energy minimization is required. Future additions will allow for MD simulations within the same toolkit.

## Contributors
* Beatriz Gonzalez del Rio (brio3@gatech.edu)
* Rampi Ramprasad

## License & copyright
Ramprasad Group, Georgia Tech, USA

Licensed under the [GTRC License](LICENSE). 

## Installation
ML-DFT requires the following packages to be installed in order to function properly:
* python 3.7
* joblib 0.13.2
* pandas
* scipy 1.3.1
* matplotlib
* scikit-learn 0.21.3
* keras 2.2.4
* tensorflow 1.14
* pymatgen 2019.10.4
* h5py 2.10.0 (may need to do: pip install h5py==2.10.0 --force-reinstall)


We recommend using Anaconda python, and creating a fresh conda environment for ML-DFT (e. g. `conda create -n MY_ENV_NAME`).
Use the file cpu_mldft.txt file to clone the environment on any cpu machine. It includes all the packages. 
Once all necessary packages are installed, clone the ML-DFT repository and install it using the *setup.py* included in the package.

```angular2
python setup.py install
```
To run the package for prediction and training follow QUICK_GUIDE.md
