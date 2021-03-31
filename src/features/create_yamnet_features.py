# -*- coding: utf-8 -*-

"""YAMnet feature creation script."""

import os

import numpy as np
from omegaconf import OmegaConf

from src.data import load_metadata, find_paths
from src.features.YAMNetExtractor import YAMNetExtractor
import src.utils


def main():
    """Main method when run as script."""
    # Load the configuration
    conf = OmegaConf.load("./config.yaml")

    # Setup the GPUs to use
    src.utils.gpu_setup()

    # Load the metadata and get the subset to use
    metadata = load_metadata()
    uri_list = np.loadtxt(conf.uri_path, dtype=str)
    sel = [uri in uri_list for uri in metadata.episode_uri]
    subset = metadata.iloc[sel]

    # Generate the input and output paths
    output_path = os.path.join(conf.output_path, "yamnet")
    print("Taking input from {}".format(conf.data_audio))
    print("Extracting output to {}".format(output_path))
    input_paths = find_paths(subset, conf.data_audio, ".ogg")
    embed_paths = find_paths(subset, os.path.join(output_path, "embedding"), ".h5")
    output_paths = find_paths(subset, os.path.join(output_path, "scores"), ".h5")

    # Run the YAMnet feature extraction
    ex = YAMNetExtractor()
    ex.embedding(input_paths, output_paths, embed_paths)


if __name__ == "__main__":
    main()
