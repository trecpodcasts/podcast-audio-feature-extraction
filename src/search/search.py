# -*- coding: utf-8 -*-

"""Script utilities."""

import os
import requests
import argparse
import pickle

from omegaconf import OmegaConf
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from tensorflow.keras.models import load_model
import soundfile as sf
import opensmile
import resampy
import laugh_segmenter  # noqa: E402
import params as yamnet_params  # noqa: E402
import yamnet as yamnet_model  # noqa: E402

from src.data import load_metadata, find_paths
import src.utils


class Searcher:
    """Class to run search queries with."""

    def __init__(self, config_path="./config.yaml"):
        """Init method for the Searcher."""
        super().__init__()
        # Load the configuration
        conf = OmegaConf.load(config_path)
        self.dataset_path = conf.dataset_path
        self.audio_path = os.path.join(conf.dataset_path, "podcasts-audio")

        self.es_url = conf.search_es_url  # URL of Elasticsearch to query
        self.es_num = conf.search_es_num  # Number of segments to request from Elasticsearch
        self.sample_rate = 44100  # Hardcoded sample rate of all podcast audio

        # Set up the reranking model
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            conf.search_rerank_model, use_fast=True, cache_dir=conf.search_cache_dir
        )
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            conf.search_rerank_model, cache_dir=conf.search_cache_dir
        )
        self.rerank_model.to("cpu", non_blocking=True)
        self.rerank_max_seq_len = 512

        # Set up the emotion model
        self.emotion_tokenizer = T5Tokenizer.from_pretrained(
            conf.search_emotion_model, cache_dir=conf.search_cache_dir
        )
        self.emotion_model = T5ForConditionalGeneration.from_pretrained(
            conf.search_emotion_model, cache_dir=conf.search_cache_dir
        )

        # Set up the openSMILE extractor
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            options={
                "frameModeFunctionalsConf": os.path.join(
                    os.getenv("PODCAST_PATH"),
                    "data/custom_FrameModeFunctionals.conf.inc",
                )
            },
        )

        # Set up the YAMNet model
        params = yamnet_params.Params(
            sample_rate=self.sample_rate, patch_hop_seconds=0.48
        )
        self.yamnet_classes = yamnet_model.class_names(
            os.path.join(os.getenv("YAMNET_PATH"), "yamnet_class_map.csv")
        )
        self.yamnet_model = yamnet_model.yamnet_frames_model(params)
        self.yamnet_model.load_weights(
            os.path.join(os.getenv("PODCAST_PATH"), "data/yamnet.h5")
        )

        # Set ip the LaughterDetection model
        self.laughter_model = load_model(
            os.path.join(os.getenv("LAUGHTER_PATH"), "models/model.h5")
        )

    def search(self, topic_query, topic_desc=None, mood="entertaining", num=10):
        """Run the full search and reranking pipeline for a query."""
        # 1) Run the basic Elasticsearch query on the segments index
        segments, es_scores = self.elasticsearch_query(topic_query, topic_desc)

        # 2) Get the BERT(MS MARCO) rerank scores for the segments
        rerank_scores = self.get_rerank_scores(segments, topic_query, topic_desc)

        # 3) Get the T5(GoEmotions) emotion scores for the segments
        emotion_scores = self.get_emotion_scores(segments)

        # 4) Run a rerank now, and only compute audio features on interesting segments
        segments_slim = [seg for i, seg in enumerate(segments) if rerank_scores[i] > 0.0]
        es_scores_slim = [score for i, score in enumerate(es_scores) if rerank_scores[i] > 0.0]
        rerank_scores_slim = [score for score in rerank_scores if score>0.0]
        emotion_scores_slim = [score for i, score in enumerate(emotion_scores) if rerank_scores[i] > 0.0]
        
        # 5) Get the audio features for the segments
        audio_features = self.get_audio_features(segments_slim)

        # 6) Combine all the scores
        if mood == "entertaining":
            ranked_segments = self.rerank_entertaining(
                segments_slim, rerank_scores_slim, emotion_scores_slim, audio_features
            )
        elif mood == "subjective":
            ranked_segments = self.rerank_subjective(
                segments_slim, rerank_scores_slim, emotion_scores_slim, audio_features
            )
        elif mood == "discussion":
            ranked_segments = self.rerank_discussion(
                segments_slim, rerank_scores_slim, emotion_scores_slim, audio_features
            )
        else:
            raise ValueError("Not a valid mood query!")

        return ranked_segments

    def elasticsearch_query(self, topic_query, topic_desc=None):
        """Run a topical search query on the Elasticsearch index."""
        print("Running Elasticsearch query...")
        # TODO: Look at a more complex query using the "topic_desc"
        json = {
            "size": self.es_num,
            "query": {
                "multi_match": {
                    "query": topic_query,
                    "fields": ["seg_words", "epis_name", "epis_desc"],
                }
            },
        }
        # Attempt the Elasticsearch query
        try:
            response = requests.get(url=self.es_url, json=json)
        except Exception as e:
            raise ConnectionError("Could not connect to Elasticsearch.")
        # Unpack the response segments and scores and return
        response = response.json()["hits"]["hits"]
        segments = []
        for seg in response:
            seg_dict = seg["_source"]
            seg_dict["seg_id"] = seg["_id"]
            segments.append(seg_dict)
        scores = np.array([seg["_score"] for seg in response])
        return segments, scores

    def elasticsearch_test(self, num_segments=10):
        """Can be used to run the full reranking pipeline without Elasticsearch."""
        # Should just load a data pickle file of the response for the trump query
        with open(
            os.path.join(os.getenv("PODCAST_PATH"), "data/test_es.pkl"), "rb"
        ) as f:
            save_dict = pickle.load(f)
        return save_dict["segments"][:num_segments], save_dict["scores"][:num_segments]

    def get_rerank_scores(self, segments, topic_query, topic_desc=None):
        """Get the rerank model scores for the given segments."""
        print("Getting rerank scores for segments...")
        # Create the topic query and description input
        if topic_desc:
            topic_input = topic_query + " " + topic_desc
        else:
            topic_desc = topic_query

        # Create the choices to rank against, we use "seg_words" and "epis_desc" fields
        choices = [
            choice["seg_words"] + " " + choice["epis_desc"] for choice in segments
        ]

        # Tokenise the topic_input and choice pairs
        inputs = [
            self.rerank_tokenizer.encode_plus(
                topic_input,
                choice,
                add_special_tokens=True,
                return_token_type_ids=True,
                truncation=True,
                max_length=self.rerank_max_seq_len,
            )
            for choice in choices
        ]

        max_len = min(max(len(t["input_ids"]) for t in inputs), self.rerank_max_seq_len)
        input_ids = [
            t["input_ids"][:max_len] + [0] * (max_len - len(t["input_ids"][:max_len]))
            for t in inputs
        ]
        attention_mask = [
            [1] * len(t["input_ids"][:max_len])
            + [0] * (max_len - len(t["input_ids"][:max_len]))
            for t in inputs
        ]
        token_type_ids = [
            t["token_type_ids"][:max_len]
            + [0] * (max_len - len(t["token_type_ids"][:max_len]))
            for t in inputs
        ]

        input_ids = torch.tensor(input_ids).to("cpu", non_blocking=True)
        attention_mask = torch.tensor(attention_mask).to("cpu", non_blocking=True)
        token_type_ids = torch.tensor(token_type_ids).to("cpu", non_blocking=True)

        # Run through the reranking model to get associated scores
        with torch.no_grad():
            logits = self.rerank_model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )[0]
            logits = logits.detach().cpu().numpy()

        # We just return the second number in the logits, may want to check this
        return np.array([score[1] for score in logits])

    def get_emotion_scores(self, segments):
        """Get the emotion scores for the given segments."""
        print("Getting emotion scores for segments...")
        # Run each segment through the emotion model
        emotions = []
        for seg in segments:
            input_text = "emotion: {}".format(seg["seg_words"])
            features = self.emotion_tokenizer([input_text], return_tensors="pt")
            tokens = self.emotion_model.generate(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
                max_length=64,
            )
            emotions.append(
                self.emotion_tokenizer.decode(tokens[0], skip_special_tokens=True)
            )
        return np.array(emotions)

    def get_paths_and_starts(self, segments):
        """Get the audio file paths for the given segments."""
        paths = []
        starts = []
        metadata = load_metadata(self.dataset_path)
        for segment in segments:
            episode_uri, segment_start = segment["seg_id"].split("_")
            episode_uri = "spotify:episode:" + episode_uri
            episode = metadata[(metadata.episode_uri == episode_uri).values]
            audio_file = find_paths(episode, self.audio_path, ".ogg")
            paths.append(audio_file[0])
            starts.append(int(segment_start))
        return paths, starts

    def get_audio_segment(self, path, start, duration=120):
        """Get the required podcast audio segment."""
        waveform, sr = sf.read(
            path,
            start=start * self.sample_rate,
            stop=(start + duration) * self.sample_rate,
            dtype=np.int16,
        )
        if sr != self.sample_rate:
            raise ValueError("Sample rate does not match the required value.")
        waveform = np.mean(waveform, axis=1) / 32768.0
        return waveform

    def get_laughter_scores(self, waveform, threshold=0.5, min_length=0.2):
        # Resample waveform to sr=8000
        y_low = resampy.resample(waveform, self.sample_rate, 8000)

        # Create the features and run through laughter model
        feature_list = laugh_segmenter.get_feature_list(y_low, 8000)
        probs = self.laughter_model.predict(feature_list)
        probs = probs.reshape((len(probs),))
        filtered = laugh_segmenter.lowpass(probs)
        instances = laugh_segmenter.get_laughter_instances(
            filtered, threshold=threshold, min_length=min_length
        )

        if len(instances) > 0:
            return [{"start": i[0], "end": i[1]} for i in instances]
        else:
            return []

    def get_audio_scores(self, segments):
        """Get the audio features for the given segments.

        Here we get the LaughterDetection, YAMNet, and openSMILE features for the segments.
        """
        print("Getting audio scores for segments...")
        # Get the audio paths for the segments
        paths, starts = self.get_paths_and_starts(segments)

        # Run each segment through each of the audio models and feature extractors
        smile_score_list, yamnet_scores_list, laughter_scores_list = [], [], []
        for path, start in zip(paths, starts):
            # Get the segment audio
            waveform = self.get_audio_segment(path, start)

            # Run through openSMILE for eGeMAPS features
            opensmile_features = self.smile.process_signal(waveform, self.sample_rate)
            smile_score_list.append(opensmile_features)

            # Run through the YAMNet model
            yamnet_scores, embeddings, spectrogram = self.yamnet_model(waveform)
            yamnet_scores_list.append(yamnet_scores.numpy())

            # Run through the LaughterDetection model
            laughter_scores = self.get_laughter_scores(waveform)
            laughter_scores_list.append(laughter_scores)

        return smile_score_list, yamnet_scores_list, laughter_scores_list

    def rerank_entertaining(
        self, segments, rerank_scores, emotion_scores, audio_features
    ):
        # TODO: Probably want to rerank in bands as Jussi suggested
        # TODO: Can do lots of rejection on things we don't want, music, quality, num speakers
        # TODO: Need to remember that LaughterDetection gets confused with music
        # rerank_ranks = np.argsort(rerank_scores_filter)[::-1]
        raise NotImplementedError

    def rerank_subjective(
        self, segments, rerank_scores, emotion_scores, audio_features
    ):
        # TODO: Probably want to rerank in bands as Jussi suggested
        # TODO: Can do lots of rejection on things we don't want, music, quality, num speakers
        # TODO: Need to remember that LaughterDetection gets confused with music
        raise NotImplementedError

    def rerank_discussion(
        self, segments, rerank_scores, emotion_scores, audio_features
    ):
        # TODO: Probably want to rerank in bands as Jussi suggested
        # TODO: Can do lots of rejection on things we don't want, music, quality, num speakers
        # TODO: Need to remember that LaughterDetection gets confused with music
        raise NotImplementedError


def print_segments(segments):
    """Pretty-print the search response segments."""
    for seg in segments:
        print(
            "Segment ID: {}, Words: {}, Show Name: {}, Episode Name: {}".format(
                seg["seg_id"],
                seg["seg_length"],
                seg["show_name"],
                seg["epis_name"],
            )
        )
        print("\n{}\n\n".format(seg["seg_words"]))


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Performs a topical segment search")
    parser.add_argument("query", help="search query")
    parser.add_argument("--desc", help="search description", default=None)
    parser.add_argument("-n", "--num", help="Number of results to return", default=10)
    return parser.parse_args()


def main():
    """Run a search query."""
    args = parse_args()
    if not args:
        raise ValueError("Invalid Arguments")

    # Setup the GPUs
    src.utils.gpu_setup()

    # Setup the searcher
    searcher = Searcher()

    # Run the query
    results = searcher.search(args.query, args.desc, args.num)

    # Print the results
    print(results)


if __name__ == "__main__":
    main()
