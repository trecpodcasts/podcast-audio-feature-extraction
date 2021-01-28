__all__ = [ "load_metadata", "find_file_paths", "load_transcript", 
            "retrieve_full_transcript", "load_audio" ]

DATA_PATH = "/unix/cdtdisspotify/data/spotify-podcasts-2020/"

import json
import os
import numpy as np
import pandas as pd 


def load_metadata():
    return pd.read_csv(DATA_PATH + "metadata.tsv", delimiter='\t')

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
    relative_file_path = os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix
    )
    
    transcript_path = os.path.join(
        DATA_PATH,
        "podcasts-transcripts/",
        relative_file_path + ".json"   
    )
    
    audio_path = os.path.join(
        DATA_PATH,
        "podcasts-audio/",
        relative_file_path + ".ogg"   
    )
    
    return transcript_path, audio_path


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
