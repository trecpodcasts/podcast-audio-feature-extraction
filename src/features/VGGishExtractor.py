# -*- coding: utf-8 -*-

"""VGGish feature extractor."""

import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf  # only tf.v1 in this function
from functools import partial
import vggish_input  # noqa: E402
import vggish_slim  # noqa: E402
import vggish_params  # noqa: E402
import vggish_postprocess  # noqa: E402

from src.features import FeatureExtractor

tf.disable_v2_behavior()


class VGGishExtractor(FeatureExtractor):
    """Class for feature extraction with VGGish.

    example:
    ex = VGGishExtractor()
    ex.pre_processing(input_paths, output_paths)
    """

    def __init__(self, logfile="./log_vggish"):
        """Init method for VGGishExtractor."""
        super().__init__(logfile=logfile)
        self.model_checkpoint = os.path.join("./data/vggish_model.ckpt")
        self.pca_parameters = os.path.join("./data/vggish_pca_params.npz")

    def pre_processing(self, input_paths, output_paths, num_workers=1):
        """Run VGGish preprocessing."""
        paths = list(zip(input_paths, output_paths))
        self.multi_process(self._pre_process, paths, num_workers)

    @staticmethod
    def _pre_process(paths):
        """Individual VGGish preprocessing process."""
        input_path, output_path = paths
        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(
            input_path, output_path
        )

        if input_path_exists and not output_path_exists:
            features = vggish_input.wavfile_to_examples(
                input_path
            )  # can also do .ogg files
            pickle.dump(features, open(output_path, "wb"))
            del features

    def embedding(self, input_paths, output_paths):
        """Run VGGish embedding."""
        paths = list(zip(input_paths, output_paths))

        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, self.model_checkpoint)

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME
            )
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME
            )

            func = partial(
                self._embed,
                sess=sess,
                features_tensor=features_tensor,
                embedding_tensor=embedding_tensor,
            )

            self.single_process(func, paths)

    @staticmethod
    def _embed(paths, sess, features_tensor, embedding_tensor):
        """Individual VGGish embedding process."""
        input_path, output_path = paths
        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(
            input_path, output_path
        )

        if input_path_exists and not output_path_exists:

            log_mel = pickle.load(open(input_path, "rb"))

            embedding = np.zeros((log_mel.shape[0], 128))

            size = len(log_mel)
            i = 0
            di = 100
            while i <= size:

                [embedding_batch] = sess.run(
                    [embedding_tensor], feed_dict={features_tensor: log_mel[i : i + di]}
                )

                embedding[i : i + di] = embedding_batch
                i += di

            pickle.dump(embedding, open(output_path, "wb"))
            del embedding

    def post_processing(self, input_paths, output_paths, num_workers=1):
        """Run VGGish postprocessing."""
        paths = list(zip(input_paths, output_paths))
        post_processor = vggish_postprocess.Postprocessor(self.pca_parameters)
        func = partial(self._post_process, post_processor=post_processor)
        self.multi_process(func, paths, num_workers=num_workers)

    @staticmethod
    def _post_process(paths, post_processor):
        """Individual VGGish postprocessing process."""
        input_path, output_path = paths
        input_path_exists, output_path_exists = FeatureExtractor.feature_path_checker(
            input_path, output_path
        )

        if input_path_exists and not output_path_exists:
            embedding = pickle.load(open(input_path, "rb"))
            postprocessed = post_processor.postprocess(embedding)
            pickle.dump(postprocessed, open(output_path, "wb"))

            del postprocessed
            del embedding
