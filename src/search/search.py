import sys
import requests
import argparse


def print_response(response):
    """Pretty-print the search response hits."""
    for hit in response.json()["hits"]["hits"]:
        print(
            "Score: {}, Segment ID: {}, Words: {}, Show Name: {}, Episode Name: {}".format(
                hit["_score"],
                hit["_id"],
                hit["_source"]["seg_length"],
                hit["_source"]["show_name"],
                hit["_source"]["epis_name"],
            )
        )
        print("{}\n".format(hit["_source"]["seg_words"]))


def search(query, to_return=3, boost=False):
    """Run a topical search request."""
    # TODO: Implement a multi_match query including the show and episode names and descriptions
    url = "http://localhost:9200/segments/_search"
    json = {"size": to_return, "query": {"match": {"seg_words": query}}}

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


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Creates batch farm submission scripts"
    )
    parser.add_argument("query", help="search query")
    parser.add_argument("-n", "--num", help="Number of results to return", default=3)
    parser.add_argument(
        "--boost", action="store_true", help="Should we boost the results with model?"
    )
    return parser.parse_args()


def main():
    """Main method when run as script."""
    args = parse_args()
    if not args:
        print("Invalid Arguments")
        sys.exit(1)
    search(args.query, args.num, args.boost)


if __name__ == "__main__":
    main()
