# podcast-dataset

Spotify Podcast Dataset software

## Usage

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