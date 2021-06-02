# podcast-dataset

Spotify Podcast Dataset software

SHOULD ADD LINK TO REPORT AND PAPER!!!

## Getting started

Initialise submodules and install dependencies/src into a conda environment:

```bash
cd podcast-dataset
git submodule update --init --recursive
conda env create -f environment.yaml
conda activate podcasts
pip install -e .
```

Setup the environment variables, adding the paths for the VGGish/YAMNet code:

```bash
source env_vars.sh
```

Download the required model files:

```bash
curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -o data/vggish_pca_params.npz https://storage.googleapis.com/audioset/vggish_pca_params.npz
curl -o data/yamnet.h5 https://storage.googleapis.com/audioset/yamnet.h5
```

## Generating features

To generate features, first, modify the config.yaml configuration file for your setup and then run one of the extraction scripts in ./src/features/. For example, to generate the eGeMAPS features run:

```bash
python src/features/create_opensmile_features.py
```

## Segment Retrieval

Segment retrieval requires a running Elasticsearch instance at http://localhost:9200. You can download Elasticsearch from https://www.elastic.co/downloads/elasticsearch.

First, all segment must be indexed into an Elasticsearch index. Modify the config.yaml configuration file for your setup and then run:

```bash
source src/search/index.py
```

You can then run segment retrieval, which will return the baseline textual reranked segments alongside the audio-enhanced ones:

```bash
source src/search/search.py [topical query title] --desc [topical query description] -n [number of segments to retrieve]
```

## Notebooks

The ./notebooks directory contains some jupyter notebooks we hope will be helpful:

- search.ipynb gives examples from the segment retrieval procedure

