__all__ = [ "load_metadata", "find_file_paths", "find_file_paths_features",
"load_transcript", "retrieve_full_transcript", "load_audio" ]

DATA_PATH = "/unix/cdtdisspotify/data/spotify-podcasts-2020/"
DATA_PATH_GPU02 = "/mnt/storage/cdtdisspotify/eGeMAPSv01b/intermediate/"


import json
import os
import numpy as np
import pandas as pd 


def load_metadata():
    return pd.read_csv(DATA_PATH + "metadata.tsv", delimiter='\t')


def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """ returns the relative filepath based on the episode metadata """
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix
    )

def find_file_paths(show_filename_prefix, episode_filename_prefix):
    """
    input:
        - show_filename_prefix -> as given in metadata-*.tsv
        - episode_filename_prefix -> as given in metadata-*.tsv
     
    returns:
        - path to transcript .json file
        - path to audio .ogg file
        
    # TODO also include the sets for summarization-testset
    """

    relative_path = relative_file_path(
        show_filename_prefix, 
        episode_filename_prefix
        )

    transcript_path = os.path.join(
        DATA_PATH,
        "podcasts-transcripts/",
        relative_path + ".json"   
    )
    
    audio_path = os.path.join(
        DATA_PATH,
        "podcasts-audio/",
        relative_path + ".ogg"   
    )
    
    return transcript_path, audio_path

def find_file_paths_features(show_filename_prefix, episode_filename_prefix):
    """
    input:
        - show_filename_prefix -> as given in metadata-*.tsv
        - episode_filename_prefix -> as given in metadata-*.tsv
     
    returns:
        - path to .pkl feature file on the local storage of the gpu02 machine
        
    """
    relative_path = relative_file_path(show_filename_prefix, episode_filename_prefix)
    
    feature_path = os.path.join(
        DATA_PATH_GPU02,
        relative_path + ".pkl"   
    )
    
    return feature_path


def load_transcript(path):
    """
    returns a python dictionary with the .json transcript
    """
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript


def retrieve_full_transcript(transcript_json):
    """
    returns the full transcript without timestamps or speakertags
    """
    
    transcript = ""
    for result in transcript_json["results"][:-1]:
        transcript += result["alternatives"][0]['transcript']
    return transcript


def load_audio(path):
    # TODO implement this feature
    raise NotImplementedError
