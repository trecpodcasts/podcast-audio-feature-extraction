# -*- coding: utf-8 -*-

"""openSMILE feature creation script."""

import os

import numpy as np
from omegaconf import OmegaConf

from src.data import load_metadata, find_paths
from src.features.OpenSmileExtractor import OpenSmileExtractor


def main():
    """Run the openSMILE feature extraction."""
    # Load the configuration
    conf = OmegaConf.load("./config.yaml")

    # Load the metadata and get the subset to use
    metadata = load_metadata(conf.dataset_path)
    uri_list = np.loadtxt(conf.features_uri_path, dtype=str)
    sel = [uri in uri_list for uri in metadata.episode_uri]
    subset = metadata.iloc[sel]

    # Generate the input and output paths
    input_path = os.path.join(conf.dataset_path, "podcasts-audio")
    output_path = os.path.join(conf.features_output_path, "opensmile")
    print("Taking input from {}".format(input_path))
    print("Extracting output to {}".format(output_path))
    input_paths = find_paths(subset, input_path, ".ogg")
    output_paths = find_paths(subset, output_path, ".h5")

    # Run the openSMILE feature extraction
    ex = OpenSmileExtractor(
        logfile=os.path.join(conf.features_output_path, "log_opensmile"),
    )
    ex.extract(input_paths, output_paths, conf.features_num_workers)


if __name__ == "__main__":
    main()
