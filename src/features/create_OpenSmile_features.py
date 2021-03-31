import numpy as np

from src.data import load_metadata, find_paths
from src.data import DATA_AUDIO, DATA_OPENSMILE_FUNCTIONALS,


## Change these for your machine:

# the base folder of the audio data, where you have the folder 0/, .., 7/
DATA_AUDIO = DATA_AUDIO 
# the base folder where you want the embedding saved
DATA_OPENSMILE_FUNCTIONALS = DATA_OPENSMILE_FUNCTIONALS 


def run_opensmile(metadata, num_workers=1):
    from src.features import OpenSmileExtractor
    
    input_paths = find_paths(metadata, DATA_AUDIO, ".ogg")
    output_paths = find_paths(metadata, DATA_OPENSMILE_FUNCTIONALS, ".h5")
    
    ex = OpenSmileExtractor()
    ex.extract(input_paths, output_paths, num_workers=num_workers)


metadata = load_metadata()
uri_list = np.loadtxt("./uri_list2.txt", dtype=str)
sel = [uri in uri_list for uri in metadata.episode_uri]
subset = metadata.iloc[sel]
num_workers=25

run_opensmile(subset, num_workers=2)