# podcast-dataset

## Usage

It is best to use the UCL HEP GPU machines for all work as they are powerful and come equipped with multiple GPUs to dramatically speed up certain computations. To use one of the GPU machines you must first ssh into one of the HEP login nodes and then ssh from there into one of the GPU machines. To do this run the following, with [username] replaced with your username...

```bash
ssh [username]@plus1.hep.ucl.ac.uk  # Can be plus2.hep.ucl.ac.uk if you want
ssh [username]@gpu01  # Can be either gpu01, gpu02, or gpu03
```

Each GPU machine has access to a local storage area (a physical disk on the machine) at /mnt/storage which may be needed for training data if loading it over the UCL HEP network from /unix/cdtdisspotify takes too long. But for now, it is probably best to work from a personal directory (which can be created) within /unix/cdtdisspotify.

A basic [Conda](https://docs.conda.io/en/latest/) environment (called podcast-dataset) containing all the currently required packages (as defined in "environment.yaml") is already installed and ready to use at "/unix/cdtdisspotify/env". To use this environment it must be activated using...

```bash
source /unix/cdtdisspotify/cuda_setup.sh
source /unix/cdtdisspotify/env/bin/activate
conda activate podcast-dataset
```

The "cuda_setup.sh" script is required to set up the correct [CUDA](https://developer.nvidia.com/CUDA-zone) driver version to be able to use GPUs with Tensorflow on the UCL GPU machines. For testing its probably best to create a personal copy of this environment in your directory so you can add other packages. With the podcast-dataset environment activated, this can be created and then activated using...

```bash
conda create --prefix /unix/cdtdisspotify/[username]/my-environment --clone podcast-dataset
conda activate /unix/cdtdisspotify/[username]/my-environment 
```

For all subsequent logins to the GPU machine, you can then activate your environment using...

```bash
source /unix/cdtdisspotify/cuda_setup.sh
source /unix/cdtdisspotify/env/bin/activate
conda activate /unix/cdtdisspotify/[username]/my-environment 
```

Most python packages can be installed with Conda using...
```bash
conda install librosa  # for example
```

But if not you can install them using pip
```bash
pip install librosa
```

A simple python script (run.py) contains the code to setup the GPUs for use with Tensorflow and load a configuration (./config/config.yaml) using [Hydra](https://hydra.cc/) from file.

## File structure

(based on https://drivendata.github.io/cookiecutter-data-science/)

To keep things organised we use the following structure for our code. A typical workflow would be:

1. Experiment with notebooks in `/notebooks/<notebook-name>.ipynb` 
2. Refactor useful code from notebooks into the right files of our software package which is structured as follows:

```
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │
│   ├── features       <- Scripts to turn raw data into features for modelling
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
```

## Installing the software package

Install the software package with

```bash
pip install -e .
``` 
to install the software package in developer mode. The package can be imported using the package `src`. 

Or even better: install using

```bash
pip install -r requirements.txt
```

which also installs all required packages for your code to run. 

## Using Jupyter Remotely

For exploratory work, using a Python Jupyter notebook is probably easiest. As we are running everything on a remote machine this is made slightly more difficult as you need to forward the port that Jupyter runs on. Also as multiple people sometimes use the machines Jupyter can start on different ports. The best thing to do is to add the following entry to your local ~/.ssh/config file, with [chosen_port] being a random port you choose. It's probably best to choose one in the 8890-9010 range as these are common Jupyter ports.

```
Host gpu01
    HostName gpu01
    User [username]
    ProxyJump [username]@plus1.hep.ucl.ac.uk
    LocalForward [chosen_port] 127.0.0.1:[chosen_port]
    ForwardX11 yes
    ForwardX11Trusted yes
    ForwardAgent yes
```

This then allows you to ssh directly into the GPU machine (skipping the plus1 machine) and forward a port that can then be used for Jupyter. To login from your local machine, activate your Conda environment, and open Jupyter-lab the following commands are used...

```bash
ssh gpu01
source /unix/cdtdisspotify/env/bin/activate
conda activate /unix/cdtdisspotify/[username]/my-environment
cd /unix/cdtdisspotify/[username]
jupyter-lab --port [chosen_port]
```

You should then be able to directly click on the link that Jupyter prints to the terminal and it should load in your local browser.