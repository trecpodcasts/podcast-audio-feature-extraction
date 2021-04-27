# -*- coding: utf-8 -*-

"""Script utilities."""

import os
import time
import requests
import argparse
import pickle
from pprint import pprint
import json

import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import soundfile as sf
import opensmile
import params as yamnet_params
import yamnet as yamnet_model

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

        # Load the podcast metadata
        self.metadata = load_metadata(self.dataset_path)

        # Set up the reranking model
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            conf.search_rerank_model, use_fast=True, cache_dir=conf.search_cache_dir
        )
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            conf.search_rerank_model, cache_dir=conf.search_cache_dir
        )
        self.rerank_model.to("cpu", non_blocking=True)
        self.rerank_max_seq_len = 512

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

    def search(self, topic_query, topic_desc=None):
        """Run the full search and reranking pipeline for a query."""
        # 1) Run the basic Elasticsearch query on the segments index
        segments, es_scores = self.elasticsearch_query(topic_query, topic_desc)

        # 2) Get the BERT(MS MARCO) rerank scores for the segments
        rerank_scores = self.get_rerank_scores(segments, topic_query, topic_desc)

        # 3) Run a rerank now, and only compute audio features on interesting segments
        segments_slim = [seg for i, seg in enumerate(segments) if rerank_scores[i] > 0.0]
        es_scores_slim = np.array([score for i, score in enumerate(es_scores) if rerank_scores[i] > 0.0])
        rerank_scores_slim = np.array([score for score in rerank_scores if score>0.0])
        
        # 4) Get the audio features for the segments
        opensmile_scores, yamnet_scores = self.get_audio_scores(segments_slim)

        # 5) Create the full search dataframe to make ranking decision on
        search_df = pd.DataFrame(segments_slim)
        search_df["es_score"] = es_scores_slim
        search_df["rerank_score"] = rerank_scores_slim
        search_df["opensmile_scores"] = opensmile_scores
        search_df["yamnet_scores"] = yamnet_scores

        # 6) Calculate some simple frequency features from the YAMNet scores
        search_df["laughter_freq"] = np.array([src.search.yamnet_freq_feature(seg_scores, 13) for seg_scores in search_df["yamnet_scores"]])
        search_df["music_freq"] = np.array([src.search.yamnet_freq_feature(seg_scores, 132) for seg_scores in search_df["yamnet_scores"]])
        search_df["conversation_freq"] = np.array([src.search.yamnet_freq_feature(seg_scores, 2) for seg_scores in search_df["yamnet_scores"]])
        search_df["narration_freq"] = np.array([src.search.yamnet_freq_feature(seg_scores, 3) for seg_scores in search_df["yamnet_scores"]])

        # 7) Calculate some more complicated YAMNet features
        search_df["yamnet_funny"] = np.array([src.search.yamnet_is_funny(seg_scores) for seg_scores in search_df["yamnet_scores"]])
        search_df["yamnet_conversation"] = np.array([src.search.yamnet_is_conversation(seg_scores) for seg_scores in search_df["yamnet_scores"]])

        # 8) Calculate some more complicated openSMILE features
        search_df["opensmile_debate"] = np.array([src.search.opensmile_is_debate(seg_scores) for seg_scores in search_df["opensmile_scores"]])
        search_df["opensmile_disapproval"] = np.array([src.search.opensmile_is_disapproval(seg_scores) for seg_scores in search_df["opensmile_scores"]])
        search_df = search_df.fillna(0.0)
        return search_df

    def rerank(self, search_df, num=5):
        rerank_dict = {}
        rerank_dict["topical"] = self.rerank_topical(search_df, num).tolist()
        rerank_dict["entertaining"] = self.rerank_entertaining(search_df, num).tolist()
        rerank_dict["subjective"] = self.rerank_subjective(search_df, num).tolist()
        rerank_dict["discussion"] = self.rerank_discussion(search_df, num).tolist()
        return rerank_dict

    def elasticsearch_query(self, topic_query, topic_desc=None):
        """Run a topical search query on the Elasticsearch index.
        
        Use the "topic_query" and "topic_desc" to search the 
        "seg_words", "epis_name", "epis_desc" fields. We boost the 
        "seg_words" by a factor of 2 as it is much more important.
        """
        print("Running Elasticsearch query... ", end = '')
        start_time = time.time()
        if topic_desc:
            topic_input = topic_query + " " + topic_desc
        else:
            topic_input = topic_query
        json = {
            "size": self.es_num,
            "query": {
                "multi_match": {
                    "query": topic_input,
                    "fields": ["seg_words^2", "epis_name", "epis_desc"],
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
        print("returned {} segments in {:.2f} seconds".format(len(scores), time.time() - start_time))
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
        """Get the rerank model scores for the given segments.
        
        Use the "query_title" and "query_desc" vs the "seg_words", "epis_name", and "epis_desc" fields.
        """
        print("Getting rerank scores for segments... ", end = '')
        start_time = time.time()
        # Create the topic query and description input
        if topic_desc:
            topic_input = topic_query + " " + topic_desc
        else:
            topic_desc = topic_query

        # Create the choices to rank against, we use "seg_words" and "epis_desc" fields
        choices = [seg["seg_words"] + " " + seg["epis_name"] + " " + seg["epis_desc"] for seg in segments]

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
        scores = np.array([score[1] for score in logits])

        # We just return the second number in the logits, may want to check this
        print("returned {} scores in {:.2f} seconds".format(len(scores), time.time() - start_time))
        return scores

    def id_to_path_and_start(self, id):
        """Get the audio file path and start for a given segment id."""
        episode_uri, segment_start = id.split("_")
        episode_uri = "spotify:episode:" + episode_uri
        episode = self.metadata[(self.metadata.episode_uri == episode_uri).values]  
        audio_file = find_paths(episode, self.audio_path, ".ogg")
        return audio_file[0], int(segment_start)

    def get_paths_and_starts(self, segments):
        """Get the audio file paths for the given segments."""
        paths = []
        starts = []
        for segment in segments:
            path, start = self.id_to_path_and_start(segment["seg_id"])
            paths.append(path)
            starts.append(start)
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

    def get_audio_scores(self, segments):
        """Get the audio features for the given segments.

        Here we get the YAMNet and openSMILE features for the segments.
        """
        print("Getting audio scores for {} segments... ".format(len(segments)))
        # Get the audio paths for the segments
        paths, starts = self.get_paths_and_starts(segments)

        # Run each segment through each of the audio models and feature extractors
        smile_score_list, yamnet_scores_list = [], []
        for path, start in zip(tqdm(paths), starts):
            try:
                # Get the segment audio
                waveform = self.get_audio_segment(path, start)

                # Run through openSMILE for eGeMAPS features
                opensmile_features = self.smile.process_signal(waveform, self.sample_rate)
                smile_score_list.append(opensmile_features)

                # Run through the YAMNet model
                yamnet_scores, embeddings, spectrogram = self.yamnet_model(waveform)
                yamnet_scores_list.append(yamnet_scores.numpy())
            except Exception:
                smile_score_list.append(None)
                yamnet_scores_list.append(None)

        return smile_score_list, yamnet_scores_list

    def rerank_topical(self, search_df, num=5):
        """Rerank the segments just on the topical rerank score."""
        ordered = search_df.sort_values("rerank_score", inplace=False, ascending=False)
        return ordered['seg_id'][:num].to_numpy()

    def rerank_entertaining(self, search_df, num=5):
        """Rerank the topical segments according to the "entertaining" mood.
        
        The segment is topically relevant to the topic description AND the topic is 
        presented in a way which the speakers intend to be amusing and entertaining to 
        the listener, rather than informative or evaluative.
        """
        # Accept podcasts with a a low music frequency score
        accepted = search_df[search_df["music_freq"] < 100]

        # Accept podcasts with a high YAMNet funny score
        accepted = accepted[accepted["yamnet_funny"] > 0.1]

        # If we accept less than required use the baseline topical rank
        num_accepted = len(accepted)
        if num_accepted < num:
            print("Not enough 'entertaining' segments found, appending topical rank...")
            topical = self.rerank_topical(search_df, num-num_accepted)
            if num_accepted > 0:
                accepted.sort_values("rerank_score", inplace=True, ascending=False)
                return np.concatenate((accepted['seg_id'], topical))
            else:
                return topical
        else:
            accepted.sort_values("rerank_score", inplace=True, ascending=False)
            return accepted['seg_id'][:num].to_numpy()

    def rerank_subjective(self, search_df, num=5):
        """Rerank the topical segments according to the "subjective" mood.
        
        The segment is topically relevant to the topic description AND the speaker or 
        speakers explicitly and clearly express a polar opinion about the query topic, 
        so that the approval or disapproval of the speaker is evident in the segment.
        """
        # Accept podcasts with a a low music frequency score
        accepted = search_df[search_df["music_freq"] < 100]

        # Accept podcasts with a low YAMNet funny score
        accepted = search_df[search_df["yamnet_funny"] < 0.1]

        # Accept podcasts with a high openSMILE debate score
        accepted = accepted[accepted["opensmile_debate"] > 10.0]

        # Accept podcasts with a high openSMILE dissaproval score
        accepted = accepted[accepted["opensmile_disapproval"] > 4.2]

        # If we accept less than required use the baseline topical rank
        num_accepted = len(accepted)
        if num_accepted < num:
            print("Not enough 'subjective' segments found, appending topical rank...")
            topical = self.rerank_topical(search_df, num-num_accepted)
            if num_accepted > 0:
                accepted.sort_values("rerank_score", inplace=True, ascending=False)
                return np.concatenate((accepted['seg_id'], topical))
            else:
                return topical
        else:
            accepted.sort_values("rerank_score", inplace=True, ascending=False)
            return accepted['seg_id'][:num].to_numpy()

    def rerank_discussion(self, search_df, num=5):
        """Rerank the topical segments according to the "discussion" mood.
        
        The segment is topically relevant to the topic description AND includes more 
        than one speaker participating with non-trivial topical contribution (e.g. mere grunts, 
        expressions of agreement, or discourse management cues ("go on", "right", "well, 
        I don't know ..." etc) are not sufficient).
        """
        # Accept podcasts with a a low music frequency score
        accepted = search_df[search_df["music_freq"] < 100]

        # Accept podcasts with a low YAMNet funny score
        accepted = search_df[search_df["yamnet_funny"] < 0.1]

        # Accept podcasts with a high openSMILE debate score
        accepted = accepted[accepted["opensmile_debate"] > 10.0]

        # Accept podcasts with a high openSMILE dissaproval score
        accepted = accepted[accepted["opensmile_disapproval"] > 4.2]

        # Accept podcasts with a low YAMNet narration score
        accepted = accepted[accepted["narration_freq"] < 100]

        # If we accept less than required use the baseline topical rank
        num_accepted = len(accepted)
        if num_accepted < num:
            print("Not enough 'discussion' segments found, appending topical rank...")
            topical = self.rerank_topical(search_df, num-num_accepted)
            if num_accepted > 0:
                accepted.sort_values("rerank_score", inplace=True, ascending=False)
                return np.concatenate((accepted['seg_id'], topical))
            else:
                return topical
        else:
            accepted.sort_values("rerank_score", inplace=True, ascending=False)
            return accepted['seg_id'][:num].to_numpy()

    def get_segment_audio(self, id):
        path, start = self.id_to_path_and_start(id)
        audio = self.get_audio_segment(path, start)
        return audio


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Performs a topical segment search")
    parser.add_argument("query", help="topical query title")
    parser.add_argument("--desc", help="topical query description", default=None)
    parser.add_argument("-n", "--num", help="Number of results to return", default=5)
    return parser.parse_args()


def main():
    """Run a search query."""
    args = parse_args()
    if not args:
        raise ValueError("Invalid Arguments")

    src.utils.gpu_setup()  # Setup the GPUs
    searcher = Searcher("./config.yaml")  # Setup the searcher
    search_df = searcher.search(args.query, args.desc)
    rerank_dict = searcher.rerank(search_df, args.num)
    pprint(rerank_dict)

    # Save the results to file...
    results_file = open("segments.json", "w")
    json.dump(rerank_dict, results_file)
    results_file.close()

if __name__ == "__main__":
    main()
