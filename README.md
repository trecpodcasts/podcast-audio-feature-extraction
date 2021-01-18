# podcast-dataset

Spotify Podcast Dataset software

## Setup

A basic conda environment (podcast-dataset) as defined in "environment.yaml" is installed at "/unix/cdtdisspotify/env" and can be activated by running the following commands...

```bash
source /unix/cdtdisspotify/cuda_setup.sh
source /unix/cdtdisspotify/env/bin/activate
conda activate podcast-dataset
```

The "cuda_setup.sh" script is required to set up the correct CUDA version in order to be able to use GPUs with Tensorflow on the UCL GPU machines. For testing its probably best to create a personal copy of this environment so you can add packages. With the podcast-dataset environment activated this can be done using...

```bash
conda create --prefix [your_copy_path] --clone podcast-dataset
```

A simple python script (run.py) contains the code to setup the GPUs for use with Tensorflow and load a configuration (./config/config.yaml) using [Hydra](https://hydra.cc/) from file.

## File structure

(based on https://drivendata.github.io/cookiecutter-data-science/)

To keep things organised we use the following structure for our code. Typical workflow would be:

1. Experiment with notebooks in `/notebooks/<notebook-name>.ipynb` 
2. Refactor useful code from notebooks into the right files of our software package which is structured as follows:

```
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
```

## Installing the software package

Install the software package with 
```
pip install -e .
``` 
to install the software package in developer mode. The package can be imported using the package `src`. 

Or even better: install using

 ```
 pip install -r requirements.txt
 ```

which also installs all required packages for your code to run. 
