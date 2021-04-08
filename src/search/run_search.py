# -*- coding: utf-8 -*-

"""Script to run a search query."""

import sys
import requests
import argparse

from src.search import Reranker


def search(query, num=10, boost=False):
    """Run a topical search request."""
    # TODO: Implement a multi_match query including the show and episode names and descriptions
    url = "http://localhost:9200/segments/_search"
    json = {"size": num, "query": {"match": {"seg_words": query}}}

    if boost:
        url = "http://localhost:8000/segments/_search"
        json["nboost"] = {
            "uhost": "localhost",
            "uport": 9200,
            "query_path": "body.query.match.seg_words",
            "topk_path": "body.size",
            "default_topk": 10,
            "topn": 50,
            "choices_path": "body.hits.hits",
            "cvalues_path": "_source.seg_words",
        }

    response = requests.get(url=url, json=json)
    print_response(response)
    return response.json()["hits"]["hits"]





def combine_scores():
    """Combine the various scores for the final segment score."""
    raise NotImplementedError


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Performs a topical segment search")
    parser.add_argument("query", help="search query")
    parser.add_argument("--desc", help="search description", default=None)
    parser.add_argument("-n", "--num", help="Number of results to return", default=10)
    return parser.parse_args()


def main():
    """Run a search query."""
    # Parse the command line arguments
    args = parse_args()
    if not args:
        print("Invalid Arguments")
        sys.exit(1)

    # Perform the base Elasticsearch query using the topic title and description
    segments = elasticsearch_query(args.query, args.desc, args.num)

    # Get the various associated scores for the segments to consider
    # TODO: Maybe add in sentiment as well
    rerank_scores = rerank_score(segments, args.query, args.desc)
    emotion_scores = get_emotion(segments)
    laughter_scores = get_laughter(segments)
    yamnet_scores = get_yamnet(segments)

    # Combine all the scores to return the final reranked segment scores
    ranked_segments = combine_scores(
        segments, 
        rerank_scores,
        emotion_scores,
        laughter_scores,
        yamnet_scores
    )

    # Reorder and print the top ranked segments.

if __name__ == "__main__":
    main()
