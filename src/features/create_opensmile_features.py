# -*- coding: utf-8 -*-

"""openSMILE feature creation script."""

import os

import numpy as np
from omegaconf import OmegaConf

from src.data.data import load_metadata, find_paths
from src.features.OpenSmileExtractor import OpenSmileExtractor


def main():
    """Main method when run as script."""
    # Load the configuration
    conf = OmegaConf.load("./config.yaml")

    # Load the metadata and get the subset to use
    metadata = load_metadata()
    uri_list = np.loadtxt(conf.uri_path, dtype=str)
    sel = [uri in uri_list for uri in metadata.episode_uri]
    subset = metadata.iloc[sel]

    # Generate the input and output paths
    output_path = os.path.join(conf.output_path, "opensmile")
    input_paths = find_paths(subset, conf.data_audio, ".ogg")
    output_paths = find_paths(subset, output_path, ".h5")

    # Run the openSMILE feature extraction
    ex = OpenSmileExtractor(conf.opensmile_config)
    ex.extract(input_paths, output_paths, conf.num_workers)


if __name__ == "__main__":
    main()