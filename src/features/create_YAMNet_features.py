import numpy as np

from src.data import load_metadata, find_paths
from src.data import DATA_AUDIO, DATA_YAMNET_EMBED, DATA_YAMNET_SCORES


## Change these for your machine:

# the base folder of the audio data, where you have the folder 0/, .., 7/
DATA_AUDIO = DATA_AUDIO
# the base folder where you want the embedding saved
DATA_YAMNET_EMBED = DATA_YAMNET_EMBED
# the base folder where you want the scores saved
DATA_YAMNET_SCORES = DATA_YAMNET_SCORES


def run_yamnet(metadata):
    from src.features import YAMNetExtractor

    input_paths = find_paths(metadata, DATA_AUDIO, ".ogg")
    embed_paths = find_paths(metadata, DATA_YAMNET_EMBED, ".h5")
    output_paths = find_paths(metadata, DATA_YAMNET_SCORES, ".h5")

    ex = YAMNetExtractor()
    ex.embedding(input_paths, output_paths, embed_paths)  # also save embeddings


metadata = load_metadata()
uri_list = np.loadtxt("./uri_list2.txt", dtype=str)
sel = [uri in uri_list for uri in metadata.episode_uri]
subset = metadata.iloc[sel]

run_yamnet(subset)
