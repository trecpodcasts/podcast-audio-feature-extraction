# podcast-dataset

Spotify Podcast Dataset software

A report with technical detail from the project which developed this software and a published paper can be found in the ./documentation directory

- Abigail Alexander, Matthijs Mars, Josh C. Tingey, Haoyue Yu, Chris Backhouse, Sravana Reddy, and Jussi Karlgren. "Audio Features, Precomputed for Podcast Retrieval and Information Access Experiments." In Proceedings of the Conference and Labs of the Evaluation Forum (CLEF), pp. 3-14. Springer, 2021.

```
@inproceedings{alexander2021audio,
  title={Audio Features, Precomputed for Podcast Retrieval and Information Access Experiments},
  author={Alexander, Abigail and Mars, Matthijs and Tingey, Josh C and Yu, Haoyue and Backhouse, Chris and Reddy, Sravana and Karlgren, Jussi},
  booktitle={Proceedings of the Conference and Labs of the Evaluation Forum (CLEF)},
  pages={3--14},
  year={2021},
  organization={Springer}
}
```
 

## Getting started

Initialise submodules and install dependencies/src into a conda environment:

```bash
cd podcast-audio-feature-extraction
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

To generate features, first, modify the variables in the config.yaml configuration file, importantly you must set "dataset_path" to the location of the Spotify podcast dataset and "features_uri_path" to a file containing a list of podcast URI's to calculate the features for. Then run one of the extraction scripts in ./src/features/. For example, to generate the eGeMAPS features, run:

```bash
python src/features/create_opensmile_features.py
```

## Segment Retrieval

Segment retrieval requires a running Elasticsearch instance at http://localhost:9200. You can download Elasticsearch from https://www.elastic.co/downloads/elasticsearch. The segment retrieval procedure calculates the audio features on the fly for the podcast segments, so the features are not required to be generated beforehand.

First, all segment must be indexed into an Elasticsearch index. Modify the config.yaml configuration file for your setup, importantly you must set "dataset_path" to the location of the Spotify podcast dataset, and then run:

```bash
python src/search/index.py
```

You can then run segment retrieval, which will return the baseline textual reranked segments alongside the audio-enhanced ones:

```bash
python src/search/search.py [topical query title] --desc [topical query description] -n [number of segments to retrieve]
```

The output segments will be saved to a "./segments.json" file

## Notebooks

The ./notebooks directory contains some jupyter notebooks we hope will be helpful:

- feature-extraction-demo.ipynb gives examples for the extraction of the features. 
- search.ipynb gives examples from the segment retrieval procedure.

