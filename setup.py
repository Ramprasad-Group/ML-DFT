import os
from setuptools import setup, find_packages
from subprocess import call


# Read the contents of your README file
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(PACKAGE_DIR, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='MLDFT',
      version='1.0.0',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      description='Predict electronic structure of polymers',
      keywords=['electron density', 'DOS', 'energy', 'forces', 'stress', 'polymer'],
      url='https://github.com/Ramprasad-Group/ML_DFT',
      author='Beatriz Gonzalez del Rio',
      author_email='brio3@gatech.edu',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GTRC License",
        "Operating System :: OS Independent",
        ],
      packages=find_packages(),
      package_data={'': ['*.joblib','*.hdf5','*model']},
      include_package_data=True,
      install_requires=['matplotlib', 'scikit-learn', 'keras', 'tensorflow', 'scipy',
                        'pandas',
                        'joblib', 'pymatgen', 'h5py'],
      zip_safe=False
      )
