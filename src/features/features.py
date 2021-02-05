__all__ = [ "extract_features_file", "extract_features_folder", "run_feature_extraction"]

import os
import concurrent.futures
import multiprocessing
import opensmile
import time 

from functools import partial
from src.data import load_metadata, find_file_paths, find_file_paths_features

def extract_features_file(
        paths,
        feature_set = opensmile.FeatureSet.eGeMAPSv01b, 
        feature_level = opensmile.FeatureLevel.Functionals,
        num_workers = 1,
        verbose = True
        ):
    """
    docstring
    """

    input_path,  output_path = paths
    base = os.path.dirname(output_path) # returns path to folder of file
    if not os.path.exists(base):
        os.makedirs(base) # create all necessary folders

    if not os.path.exists(output_path):        
        # if file not exists, set up smile to extract features
        smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=feature_level,
            options = {},
            loglevel = 1,
            logfile = "./log.log",
            num_channels = 1,
            keep_nat = False,
            num_workers=1,
            verbose=verbose
        )

        # extracting only 1 channel (should not matter much in case of podcasts)
        features = smile.process_file(input_path, channel=1) 
        features.to_pickle(output_path) # save to pickle file
        del features



def extract_features_metadata(       
        n_episodes, 
        feature_set = opensmile.FeatureSet.eGeMAPSv01b, 
        feature_level = opensmile.FeatureLevel.Functionals,
        num_workers = 1,
        verbose = False
    ):
    """
    extract the features of 'n_episodes' of podcasts in order of the metadata. 

    n_episodes: number of episodes to process in order of metadata listing
    """

    function = partial(
        extract_features_file, 
        feature_set = feature_set, 
        feature_level = feature_level,
        num_workers=1,
        verbose=verbose
    )

    metadata = load_metadata()

    paths = [ ( 
            find_file_paths(
            metadata.show_filename_prefix[i], 
            metadata.episode_filename_prefix[i]
            )[1],
            find_file_paths_features(
            metadata.show_filename_prefix[i], 
            metadata.episode_filename_prefix[i]
            ) 
            )  for i in range(n_episodes)
    ]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(function, paths)
    

def run_feature_extraction():
    print("started extracting functionals with 1s window")
    t3_start = time.perf_counter()
    new_func(
        n_episodes = 1000,
        num_workers = 30
    )
    t3_stop = time.perf_counter()
    print(f"finished functionals of {n_episodes} podcasts in:", t3_stop-t3_start)