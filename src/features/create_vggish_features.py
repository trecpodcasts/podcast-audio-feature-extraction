# -*- coding: utf-8 -*-

"""VGGish feature creation script."""

import os
import pickle
from tqdm import tqdm

import numpy as np
from omegaconf import OmegaConf

from src.data import load_metadata, find_paths
from src.features.VGGishExtractor import VGGishExtractor
import src.utils


def combine_vggish_features(metadata, base_dir, output_file="./result.pkl"):
    """combines vggish features of the first 10 min of a podcast (only works with podcast longen than 10min)"""

    input_paths = find_paths(metadata, base_dir, ".pkl")
    data = {}
    for i in tqdm(range(len(metadata))):
        if os.path.exists(input_paths[i]):
            data[metadata.episode_uri.iloc[i]] = (
                pickle.load(open(input_paths[i], "rb"))[:6000:5]
                .astype(np.float16)
                .tolist()
            )

    with open(output_file, "wb") as file:
        pickle.dump(data, file)


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
    output_path = os.path.join(conf.output_path, "vggish")
    print("Taking input from {}".format(conf.data_audio))
    print("Extracting output to {}".format(output_path))
    input_paths = find_paths(subset, conf.data_audio, ".ogg")
    output_log_mel = find_paths(subset, os.path.join(output_path, "log_mel"), ".pkl")
    output_embedding = find_paths(
        subset, os.path.join(output_path, "embedding"), ".pkl"
    )
    output_postprocessed = find_paths(
        subset, os.path.join(output_path, "postprocessed"), ".pkl"
    )

    ex = VGGishExtractor()
    ex.pre_processing(input_paths, output_log_mel, num_workers=conf.num_workers)
    ex.embedding(output_log_mel, output_embedding)
    ex.post_processing(
        output_embedding, output_postprocessed, num_workers=conf.num_workers
    )

    #combine_vggish_features(
    #    subset.iloc[:1500],
    #    output_postprocessed,
    #    "/mnt/storage/cdtdisspotify/results/yamnet_scores1.pkl",
    #)


if __name__ == "__main__":
    main()
