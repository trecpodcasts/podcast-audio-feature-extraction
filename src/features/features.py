
import os
import pickle
import numpy as np

from src.data import load_metadata, find_paths
from src.data import DATA_AUDIO, DATA_OPENSMILE_FUNCTIONALS_1s, DATA_VGGISH_LOG_MEL, DATA_VGGISH_EMBED, DATA_VGGISH_POSTPROCESSED



def run_opensmile(metadata, num_workers=1):
    from src.features import OpenSmileExtractor
    
    input_paths = find_paths(metadata, DATA_AUDIO, ".ogg")
    output_paths = find_paths(metadata, DATA_OPENSMILE_FUNCTIONALS_1s, ".pkl")
    
    ex = OpenSmileExtractor()
    ex.extract(input_paths, output_paths, num_workers=num_workers)
    
def run_vggish_preprocess(metadata, num_workers=1):
    from src.features import VGGishExtractor
    
    input_paths = find_paths(metadata, DATA_AUDIO, ".ogg")
    output_paths = find_paths(metadata, DATA_OPENSMILE_FUNCTIONALS_1s, ".pkl")
    
    ex = VGGishExtractor()
    ex.pre_processing(input_paths, output_paths, num_workers=num_workers)
    
def run_vggish_embed(metadata):
    from src.features import VGGishExtractor
    
    input_paths = find_paths(metadata, DATA_VGGISH_LOG_MEL, ".pkl")
    output_paths = find_paths(metadata, DATA_VGGISH_EMBED, ".pkl")
    
    ex = VGGishExtractor()
    ex.embedding(input_paths, output_paths)
    
def run_vggish_postprocess(metadata, num_workers=1):
    from src.features import VGGishExtractor
    
    input_paths = find_paths(metadata, DATA_VGGISH_EMBED, ".pkl")
    output_paths = find_paths(metadata, DATA_VGGISH_POSTPROCESSED, ".pkl")
    
    ex = VGGishExtractor()
    ex.post_processing(input_paths, output_paths, num_workers=num_workers)
    
def combine_vggish_features(metadata, base_dir, output_file="./result.pkl"):
    """combines vggish features of the first 10 min of a podcast (only works with podcast longen than 10min)"""
    
    input_paths = find_paths(metadata, base_dir, ".pkl")
    data = {}
    for i in range(len(metadata)):
        if os.path.exists(input_paths[i]):
            data[metadata.episode_uri.iloc[i]] = pickle.load(open(input_paths[i], "rb")).tolist()
    
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


metadata = load_metadata()
uri_list = np.loadtxt("./uri_list.txt", dtype=str)
sel = [uri in uri_list for uri in metadata.episode_uri]
subset = metadata.iloc[sel]
num_workers=25

## opensmile features:

# run_opensmile(subset, num_workers=num_workers)


## vggish features:

# run_vggish_preprocess(subset, num_workers=num_workers)
# run_vggish_embed(subset) # gpu accelerated instead of cpu
# run_vggish_postprocess(subset, num_workers=num_workers)

## combining vgg features in one file
combine_vggish_features(
    subset, 
    DATA_VGGISH_POSTPROCESSED, 
    "/mnt/storage/cdtdisspotify/results/vgg_postprocessed.pkl"
    )