# -*- coding: utf-8 -*-

"""YAMnet feature extractor."""

import os
import numpy as np
import pandas as pd
import soundfile as sf
from functools import partial
import params as yamnet_params
import yamnet as yamnet_model

from src.features import FeatureExtractor


class YAMNetExtractor(FeatureExtractor):
    """Class for feature extraction with YAMNet.

    example:
    ex = YAMNetExtractor()
    ex.embedding(input_paths, output_paths, embed_paths)
    """

    def __init__(self, logfile="./log_yamnet"):
        """Init method for YAMNetExtractor."""
        super().__init__(logfile=logfile)
        self.model_checkpoint = os.path.join("./data/yamnet.h5")
        self.class_names = os.path.join(
            os.getenv("YAMNET_PATH"), "yamnet_class_map.csv"
        )
        self.sample_rate = 44100

    def embedding(self, input_paths, output_paths, embed_paths=""):
        """Extract YAMnet features with opensmile using a single process."""
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
        """Individual YAMnet extraction process."""
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
            n_seconds = 300
            di = int(n_seconds * params.sample_rate)  # 5min segments

            real_size = 0
            while i <= waveform_size:
                scores, embeddings, spectrogram = yamnet(
                    waveform[i : i + di + int(0.47 * params.sample_rate)]
                ) # the 0.47s has to do with yamnet windowing
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
                    embed_path, "embedding", mode="w", complevel=6
                )
                del df

            df = pd.DataFrame(score, columns=class_names)
            df["time (s)"] = np.arange(len(score)) * 0.48
            df.set_index("time (s)", inplace=True)
            df.astype(np.float16).to_hdf(output_path, "score", mode="w", complevel=6)

            del df
            del embedding
            del score
            del spectrogram
