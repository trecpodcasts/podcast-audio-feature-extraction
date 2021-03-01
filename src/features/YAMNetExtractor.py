__all__ = ["YAMNetExtractor"]

import os
import sys
import pickle
import numpy as np

from functools import partial
from src.features import FeatureExtractor

YAMNET_PATH = "/mnt/storage/cdtdisspotify/models/research/audioset/yamnet/" 
assert os.path.exists(YAMNET_PATH), "The set YAMNet path cannot be found, change it in the source code"

sys.path.append(YAMNET_PATH)
import soundfile as sf
import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf

class YAMNetExtractor(FeatureExtractor):
    def __init__(self):

        self.model_checkpoint = os.path.join(YAMNET_PATH, "yamnet.h5")
        self.class_names = os.path.join(YAMNET_PATH, "yamnet_class_map.csv")
        self.sample_rate = 44100


    def embedding(self, input_paths, output_paths, embed_paths=""):
        if embed_paths == "":
            embed_paths = [""]*len(input_paths)
            save_embedding = False
        else:
            save_embedding = True

        paths = list(zip(input_paths, embed_paths, output_paths))
        
        params = yamnet_params.Params(sample_rate=self.sample_rate, patch_hop_seconds=0.1)

        class_names = yamnet_model.class_names(self.class_names)
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights(self.model_checkpoint)


        func = partial(self._embed, yamnet=yamnet, save_embedding=save_embedding)

        self.single_process(func, paths)
        

    @staticmethod
    def _embed(paths, yamnet, save_embedding=False):
        input_path, embed_path, output_path = paths
        
        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(
        input_path, output_path)


        if input_path_exists and not output_path_exists:              
            
            wav_data, sr = sf.read(input_path, dtype=np.int16)
            waveform =  np.mean(wav_data, axis=1) / 32768.0

            approx_size = int(len(waveform)/44100/0.1 ) # approximate (overestimated) size of output
            embedding = np.zeros((approx_size, 1024))
            score = np.zeros((approx_size, 521))

            waveform_size = len(waveform)
            i = 0
            di = 300 * 44100 # 5min segments

            real_size = 0
            while i <= waveform_size:

                scores, embeddings, spectrogram = yamnet(waveform[i:i+di])
                scores = scores.numpy()
                embeddings = embeddings.numpy()

                embedding[real_size: real_size + len(scores)] = embeddings
                score[real_size: real_size + len(scores)] = scores
        
                real_size += len(scores)
                i += di

            if save_embedding:

                _, _ = FeatureExtractor.feature_path_checker(
                    input_path, embed_path) # also create embed path if necessary
                pickle.dump(embedding[:real_size], open(embed_path, "wb"))
            
            pickle.dump(score[:real_size], open(output_path, "wb"))
            del embedding
            del score
            del spectrogram

