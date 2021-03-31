# -*- coding: utf-8 -*-

"""Provides helper data related functions."""

__all__ = [
    "load_metadata",
    "find_file_paths",
    "find_paths",
    "find_file_paths_features",
    "load_transcript",
    "retrieve_full_transcript",
    "retrieve_timestamped_transcript"
]

import json
import os
import warnings
import pandas as pd
import numpy as np


DATA_PATH = "/unix/cdtdisspotify/data/spotify-podcasts-2020/"
TFDS_PATH = "/mnt/storage/cdtdisspotify/tensorflow_datasets/"
DATA_PATH_GPU02 = "/mnt/storage/cdtdisspotify/eGeMAPSv01b/intermediate/"


def load_metadata():
    """Load the Spotify podcast dataset metadata."""
    return pd.read_csv(DATA_PATH + "metadata.tsv", delimiter="\t")


def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """Return the relative filepath based on the episode metadata."""
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )


def find_paths(metadata, base_folder, file_extension):
    """Finds the filepath based on the dataset structure based on the metadata, the folder where it is stored and the file extension you want.

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


def find_file_paths(show_filename_prefix, episode_filename_prefix):
    """Get the transcript and audio paths from show and episode prefix.

    Args:
        show_filename_prefix: As given in metadata-*.tsv
        episode_filename_prefix: As given in metadata-*.tsv

    Returns:
        path to transcript .json file
        path to audio .ogg file

    # TODO also include the sets for summarization-testset
    """
    warnings.warn(
        "This function is deprecated, use the find_paths function with the predefined paths in data_paths.py",
        DeprecationWarning,
    )

    relative_file_path = os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )

    transcript_path = os.path.join(
        DATA_PATH, "podcasts-transcripts/", relative_file_path + ".json"
    )
    audio_path = os.path.join(DATA_PATH, "podcasts-audio/", relative_file_path + ".ogg")
    return transcript_path, audio_path


def find_file_paths_features(show_filename_prefix, episode_filename_prefix):
    """Find the feature file on the gpu02 machine

    Args:
        show_filename_prefix: as given in metadata-*.tsv
        episode_filename_prefix: as given in metadata-*.tsv

    Returns:
        path to .pkl feature file on the local storage of the gpu02 machine
    """
    warnings.warn(
        "This function is deprecated, use the find_paths function with the predefined paths in data_paths.py",
        DeprecationWarning,
    )
    relative_path = relative_file_path(show_filename_prefix, episode_filename_prefix)

    feature_path = os.path.join(DATA_PATH_GPU02, relative_path + ".pkl")

    return feature_path


def load_transcript(path):
    """Load a python dictionary with the .json transcript"""
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript


def retrieve_full_transcript(transcript_json):
    """Load the full transcript without timestamps or speakertags"""
    transcript = ""
    for result in transcript_json["results"][:-1]:
        transcript += result["alternatives"][0]["transcript"]
    return transcript


def retrieve_timestamped_transcript(path):
    """Load the full transcript with timestamps"""
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