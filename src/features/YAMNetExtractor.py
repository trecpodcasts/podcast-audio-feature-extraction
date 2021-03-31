__all__ = ["YAMNetExtractor"]

import os
import sys
import pickle
import numpy as np
import pandas as pd

from functools import partial
from src.features import FeatureExtractor

YAMNET_PATH = "/mnt/storage/cdtdisspotify/models/research/audioset/yamnet/"
assert os.path.exists(
    YAMNET_PATH
), "The set YAMNet path cannot be found, change it in the source code"

sys.path.append(YAMNET_PATH)
import soundfile as sf
import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf


import subprocess as sp


def gpu_setup(level=7000):
    """ Script to select a free gpu on the machine"""
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    free = [True if val > level else False for val in values]
    gpus = tf.config.list_physical_devices("GPU")
    to_use = []
    for free, gpu in zip(free, gpus):
        if free:
            tf.config.experimental.set_memory_growth(gpu, True)
            to_use.append(gpu)
    try:
        tf.config.experimental.set_visible_devices(to_use, "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "actual GPUs,", len(logical_gpus), "in use.")
    except RuntimeError as e:
        print(e)


gpu_setup()


class YAMNetExtractor(FeatureExtractor):
    """Class for feature extraction with YAMNet

    example:
    ex = YAMNetExtractor()
    ex.embedding(input_paths, output_paths, embed_paths)
    """

    def __init__(self):
        super().__init__(logfile="./log_YAMNet")

        self.model_checkpoint = os.path.join(YAMNET_PATH, "yamnet.h5")
        self.class_names = os.path.join(YAMNET_PATH, "yamnet_class_map.csv")
        self.sample_rate = 44100

    def embedding(self, input_paths, output_paths, embed_paths=""):
        if embed_paths == "":
            embed_paths = [""] * len(input_paths)
            save_embedding = False
        else:
            save_embedding = True

        paths = list(zip(input_paths, embed_paths, output_paths))

        params = yamnet_params.Params(
            sample_rate=self.sample_rate, patch_hop_seconds=0.48
        )

        class_names = yamnet_model.class_names(self.class_names)
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights(self.model_checkpoint)

        func = partial(
            self._embed,
            yamnet=yamnet,
            params=params,
            class_names=class_names,
            save_embedding=save_embedding,
        )

        self.single_process(func, paths)

    @staticmethod
    def _embed(paths, yamnet, params, class_names, save_embedding=False):
        input_path, embed_path, output_path = paths

        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(
            input_path, output_path
        )

        if input_path_exists and not output_path_exists:

            wav_data, sr = sf.read(input_path, dtype=np.int16)
            waveform = np.mean(wav_data, axis=1) / 32768.0

            approx_size = int(
                len(waveform) / params.sample_rate / params.patch_hop_seconds
            )  # approximate (overestimated) size of output
            embedding = np.zeros((approx_size, 1024))
            score = np.zeros((approx_size, 521))

            waveform_size = len(waveform)
            i = 0
            di = 300 * params.sample_rate  # 5min segments

            real_size = 0
            while i <= waveform_size:

                scores, embeddings, spectrogram = yamnet(waveform[i : i + di])
                scores = scores.numpy()
                embeddings = embeddings.numpy()

                embedding[real_size : real_size + len(scores)] = embeddings
                score[real_size : real_size + len(scores)] = scores

                real_size += len(scores)
                i += di

            if save_embedding:

                _, _ = FeatureExtractor.feature_path_checker(
                    input_path, embed_path
                )  # also create embed path if necessary

                df = pd.DataFrame(embedding)
                df["time (s)"] = np.arange(len(embedding)) * 0.48
                df.set_index("time (s)", inplace=True)
                df.astype(np.float16).to_hdf(
                    embed_path, "YAMNet_Embedding", mode="w", complevel=9
                )
                del df

            df = pd.DataFrame(score, columns=class_names)
            df["time (s)"] = np.arange(len(score)) * 0.48
            df.set_index("time (s)", inplace=True)
            df.astype(np.float16).to_hdf(
                output_path, "YAMNet_Score", mode="w", complevel=9
            )

            del df
            del embedding
            del score
            del spectrogram
