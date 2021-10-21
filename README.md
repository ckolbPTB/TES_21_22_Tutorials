# TES_21_22_Tutorials
Tutorials for the Thematic Einstein Semester on Mathematics of Imaging in Real-World Challenges

The idea of the tutorials is to run them via Jupyter notebooks. 
The notebooks are designed to require little computational power. 
They can be be run either by using binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ckolbPTB/TES_21_22_Tutorials.git/HEAD)

or by using Google colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ckolbPTB/TES_21_22_Tutorials.git/HEAD/)

The two links above will open the notebooks directly in your browser and you do not need not install any additional software.
If you would like to run to code on your own computer, please follow the installation instructions below. 
If you want to do this, then please **try it well in advance to the tutorial**, because it will take some time.

## Installation

### Prerequisites 
You will need to have [git](https://www.git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [anaconda](https://docs.anaconda.com/anaconda/install/) installed on your computer. 
The notebooks do not require GPU support.

### Get the github repository
Open the terminal and clone this repository:
```
git clone https://github.com/ckolbPTB/TES_21_22_Tutorials.git
```

### Create conda environment
All the required packages are listed in `requirements.txt`. 
In order to create a conda environment with these requirements, open a terminal go to the main folder of the repository:
```
cd TES_21_22_Tutorials
```
and enter:
```
conda env create --file requirements.txt --name TES
```
Now you can activate this environment using:
```
conda activate TES
```
and start Jupyter notebooks. 
### Start notebooks

