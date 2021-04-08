# -*- coding: utf-8 -*-

"""Script utilities."""

import requests

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from src.data import load_metadata, find_paths


def TopicalSearcher():
    """Class to run search queries with."""

    def __init__(
        self,
        es_url="http://localhost:9200/segments/_search",
        es_num=100,
        cache_dir = "/unix/cdtdisspotify/transformers",
        rerank_model="amberoad/bert-multilingual-passage-reranking-msmarco",
        emotion_model="kiri-ai/t5-base-qa-summary-emotion",
        ):
        """Init method for TopicalSearcher."""
        super().__init__()
        self.es_url = es_url
        self.es_num = es_num
        self.sample_rate = 44100

        # Set up the reranking model
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model, use_fast=True, cache_dir=cache_dir)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model, cache_dir=cache_dir)
        self.rerank_model.to("cpu", non_blocking=True)

        # Set up the emotion model
        self.emotion_tokenizer = T5Tokenizer.from_pretrained(emotion_model, cache_dir=cache_dir)
        self.emotion_model = T5ForConditionalGeneration.from_pretrained(emotion_model, cache_dir=cache_dir)

        # Set up the YAMNet model
        params = yamnet_params.Params(sample_rate=self.sample_rate, patch_hop_seconds=0.48)
        class_names = yamnet_model.class_names(self.class_names)
        self.yamnet = yamnet_model.yamnet_frames_model(params)
        self.yamnet.load_weights(self.model_checkpoint)

        # Set up the laughter detection model
        

    def search(query, description=None, num=10)




def audio_snippet(id, metadata, duration=120):
    uri, start = x.split("_")
    episode = metadata[(metadata.episode_uri == uri).values]
    audio_file = find_paths(episode, DATA_AUDIO, ".ogg")
    snippet, sr = sf.read(audio_file[0], start=start*44100, stop=(start+duration)*44100, dtype=np.int16)
    return snippet, sr


def print_segments(segments):
    """Pretty-print the search response segments."""
    for seg in segments:
        print(
            "Score: {}, Segment ID: {}, Words: {}, Show Name: {}, Episode Name: {}".format(
                seg["_score"],
                seg["_id"],
                seg["_source"]["seg_length"],
                seg["_source"]["show_name"],
                seg["_source"]["epis_name"],
            )
        )
        print("{}\n".format(seg["_source"]["seg_words"]))


def elasticsearch_query(query, description=None, num=10):
    """Run a topical search query on the segments Elasticsearch index."""
    url = "http://localhost:9200/segments/_search"
    json = {"size": num, "query": {"match": {"seg_words": query}}}
    response = requests.get(url=url, json=json)
    return response.json()["hits"]["hits"]


def get_rerank(
    segments, 
    query, 
    description=None, 
    max_seq_len=512, 
    model_dir="amberoad/bert-multilingual-passage-reranking-msmarco"
    ):
    """Get the rerank model scores for the given segments."""
    # Setup the reranking model
    cache_dir = "/unix/cdtdisspotify/transformers"
    rerank_model = AutoModelForSequenceClassification.from_pretrained(model_dir, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, cache_dir=cache_dir)
    rerank_model.to("cpu", non_blocking=True)

    # Tokenise the query and choice pairs
    choices = [choice["_source"]["seg_words"] for choice in segments]
    inputs = [tokenizer.encode_plus(
        query, choice, add_special_tokens=True, return_token_type_ids=True, truncation=True, max_length=max_seq_len
        ) for choice in choices]

    max_len = min(max(len(t['input_ids']) for t in inputs), max_seq_len)
    input_ids = [t['input_ids'][:max_len] +
                    [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
    attention_mask = [[1] * len(t['input_ids'][:max_len]) +
                        [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
    token_type_ids = [t['token_type_ids'][:max_len] +
                    [0] * (max_len - len(t['token_type_ids'][:max_len])) for t in inputs]

    input_ids = torch.tensor(input_ids).to("cpu", non_blocking=True)
    attention_mask = torch.tensor(attention_mask).to("cpu", non_blocking=True)
    token_type_ids = torch.tensor(token_type_ids).to("cpu", non_blocking=True)

    # Run through the reranking model to get associated scores
    with torch.no_grad():
        logits = rerank_model(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)[0]
        logits = logits.detach().cpu().numpy()
    return logits


def get_emotion(segments):
    """Get the emotion scores for the given segments."""
    # Setup the emotion model
    model_dir = "kiri-ai/t5-base-qa-summary-emotion"
    cache_dir = "/unix/cdtdisspotify/transformers"
    tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, cache_dir=cache_dir)

    # Run each segment through the emotion model
    emotions = []
    for seg in segments:
        input_text = "emotion: {}".format(seg["_source"]["seg_words"])
        features = tokenizer([input_text], return_tensors='pt')
        tokens = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=64)
        emotions.append(tokenizer.decode(tokens[0], skip_special_tokens=True))
    return emotions


def get_laughter(segments):
    """Get the occurances of laughter for the given segments."""
    raise NotImplementedError


def get_yamnet(segments):
    """Get the YAMNet scores for the given segments."""
    raise NotImplementedError


def combine_scores(): 
    scores = []
    all_scores = []
    index_map = []
    for i, logit in enumerate(logits):
        neg_logit = logit[0]
        score = logit[1]
        all_scores.append(score)
        if score > neg_logit:
            scores.append(score)
            index_map.append(i)
    sorted_indices = [index_map[i] for i in np.argsort(scores)[::-1]]
    return logits, sorted_indices, [all_scores[i] for i in sorted_indices]