# -*- coding: utf-8 -*-

"""Data utilities."""

import json
import os
import pandas as pd
import numpy as np


def load_metadata(dataset_path):
    """Load the Spotify podcast dataset metadata."""
    return pd.read_csv(dataset_path + "metadata.tsv", delimiter="\t")


def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """Return the relative filepath based on the episode metadata."""
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )


def find_paths(metadata, base_folder, file_extension):
    """Find the filepath based on the dataset structure.

    Uses the metadata, the filepath folder and the file extension you want.

    Args:
        metadata (df): The metadata of the files you want to create a path for
        base_folder (str): base directory for where data is (to be) stored
        file_extension (str): extension of the file in the path

    Returns:
        paths (list): list of paths (str) for all files in the given metadata
    """
    paths = []
    for i in range(len(metadata)):
        relative_path = relative_file_path(
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i],
        )
        path = os.path.join(base_folder, relative_path + file_extension)
        paths.append(path)
    return paths


def load_transcript(path):
    """Load a python dictionary with the .json transcript."""
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript


def retrieve_full_transcript(transcript_json):
    """Load the full transcript without timestamps or speakertags."""
    transcript = ""
    for result in transcript_json["results"][:-1]:
        transcript += result["alternatives"][0]["transcript"]
    return transcript


def retrieve_timestamped_transcript(path):
    """Load the full transcript with timestamps."""
    with open(path, "r") as file:
        transcript = json.load(file)

    starts, ends, words, speakers = [], [], [], []
    for word in transcript["results"][-1]["alternatives"][0]["words"]:
        starts.append(float(word["startTime"].replace("s", "")))
        ends.append(float(word["endTime"].replace("s", "")))
        words.append(word["word"])
        speakers.append(word["speakerTag"])

    starts = np.array(starts, dtype=np.float32)
    ends = np.array(ends, dtype=np.float32)
    words = np.array(words)
    speakers = np.array(speakers, dtype=np.int32)
    return {"starts": starts, "ends": ends, "words": words, "speaker": speakers}
