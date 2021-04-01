# -*- coding: utf-8 -*-

"""Script to index podcast segments to Elasticsearch."""

import os
import math

import numpy as np
from tqdm import tqdm
from elasticsearch_dsl import Document, Integer, Text
from elasticsearch_dsl.connections import connections

import src.data


class PodcastSegment(Document):
    """Implementation of a podcast segment document in elasticsearch."""

    show_name = Text(analyzer="snowball")
    show_desc = Text(analyzer="snowball")
    epis_name = Text(analyzer="snowball")
    epis_desc = Text(analyzer="snowball")
    seg_words = Text(analyzer="snowball")
    seg_length = Integer()

    class Index:
        """Elasticsearch index definition."""

        name = "segments"

    def save(self, **kwargs):
        """Save the document to Elasticsearch."""
        self.seg_length = len(self.seg_words.split())
        return super(PodcastSegment, self).save(**kwargs)


def clean_text(text):
    """Clean the text to remove non-topical content.

    This includes things like episode numbers, advertisements, and links.
    """
    # TODO: Actual implement some cleaning routines
    return text


def add_podcast(
    transcript_path,
    show_name,
    show_desc,
    epis_name,
    epis_desc,
    seg_length=120,
    seg_step=60,
):
    """Get podcast transcript data to be indexed."""
    # Generate the segment basename
    seg_base = os.path.splitext(os.path.basename(transcript_path))[0] + "_"

    # Clean the show and episode names and descriptions
    show_name = clean_text(show_name)
    show_desc = clean_text(show_desc)
    epis_name = clean_text(epis_name)
    epis_desc = clean_text(epis_desc)

    # Get the transcript and find out how long it is
    transcript = src.data.retrieve_timestamped_transcript(transcript_path)
    last_word_time = math.ceil(transcript["starts"][-1])

    # Generate the segments from the start to the end of the podcasrt
    for seg_start in range(0, last_word_time, seg_step):
        # Generate the segment name
        seg_id = seg_base + str(seg_start)

        # Find the words in the segment
        word_indices = np.where(
            np.logical_and(
                transcript["starts"] >= seg_start,
                transcript["starts"] <= seg_start + seg_length,
            )
        )[0]
        seg_words = transcript["words"][word_indices]
        seg_words = " ".join(seg_words)

        # Create and save the segment
        segment = PodcastSegment(
            meta={"id": seg_id},
            show_name=show_name,
            show_desc=show_desc,
            epis_name=epis_name,
            epis_desc=epis_desc,
            seg_words=seg_words,
        )

        # Add to Elasticsearch (try 5 times)
        # TODO: Make it so the segment IDs for unindexed segements are recorded
        for i in range(5):
            try:
                segment.save()
                break
            except Exception:
                print("Retry attempt {}".format(i))


def init_index():
    """Set up the Elasticsearch index by creating the mappings."""
    PodcastSegment.init()


def main():
    """Index documents to Elasticsearch."""
    # Define client connection and setup index
    connections.create_connection(hosts=["localhost"])
    init_index()

    # For now just index the good subset of podcasts
    uri_file = open("uri_list.txt", "r")
    uri_to_use = uri_file.read()
    uri_to_use = uri_to_use.split("\n")
    uri_file.close()

    # Loop through metadata and add podcast segments to Elasticsearch
    metadata = src.data.load_metadata()
    for index, row in tqdm(metadata.iterrows()):
        if row["episode_uri"] in uri_to_use:
            transcript_path = src.data.find_file_paths(
                row["show_filename_prefix"], row["episode_filename_prefix"]
            )[0]
            add_podcast(
                transcript_path,
                row["show_name"],
                row["show_description"],
                row["episode_name"],
                row["episode_description"],
            )


if __name__ == "__main__":
    main()
