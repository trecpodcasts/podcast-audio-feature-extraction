# -*- coding: utf-8 -*-

"""Provides helper data related functions."""

__all__ = [
    "load_metadata",
    "find_file_paths",
    "load_transcript",
    "retrieve_full_transcript",
    "get_podcast_dataset",
    "get_self_supervised_data",
    "get_audio_dataset",
    "random_segment",
    "extract_log_mel_spectrogram",
]

DATA_PATH = "/unix/cdtdisspotify/data/spotify-podcasts-2020/"

import json
import os
import pandas as pd

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds


def load_metadata():
    """Load the Spotify podcast dataset metadata."""
    return pd.read_csv(DATA_PATH + "metadata.tsv", delimiter="\t")


def find_file_paths(show_filename_prefix, episode_filename_prefix):
    """Get the transcript and audio paths from show and episode prefix
    input:
        - show_filename_prefix -> as given in metadata-*.tsv
        - episode_filename_prefix -> as given in metadata-*.tsv

    returns:
        - path to transcript .json file
        - path to audio .ogg file

    # TODO also include the sets for summarization-testset
    """
    relative_file_path = os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )

    transcript_path = os.path.join(
        DATA_PATH, "podcasts-transcripts/", relative_file_path + ".json"
    )
    audio_path = os.path.join(DATA_PATH, "podcasts-audio/", relative_file_path + ".ogg")
    return transcript_path, audio_path


def load_transcript(path):
    """Load a python dictionary with the .json transcript"""
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript


def retrieve_full_transcript(transcript_json):
    """Load the full transcript without timestamps or speakertags"""
    transcript = ""
    for result in transcript_json["results"][:-1]:
        transcript += result["alternatives"][0]["transcript"]
    return transcript


def get_podcast_dataset(shuffle_buffer=1000, max_length=None):
    """Read Spotify podcast audio data from disk as tf dataset."""

    @tf.function
    def _parse_example(paths):
        audio = tfio.audio.AudioIOTensor(paths[1], dtype=tf.float32)
        audio = audio[:max_length] if max_length else audio
        return {"audio": audio / float(tf.int16.max)}

    # Get the list of (transcript, audio) paths for each episode
    metadata = load_metadata()
    show_prefixes = metadata["show_filename_prefix"]
    episode_prefixes = metadata["episode_filename_prefix"]
    paths = [find_file_paths(sp, ep) for sp, ep in zip(show_prefixes, episode_prefixes)]

    # Create the dataset, shuffle, and load data
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def get_self_supervised_data(dataset="librispeech", shuffle_buffer=1000):
    """Reads TFDS data for self-supervised task."""

    @tf.function
    def _parse_example(audio, _):
        return {"audio": tf.cast(audio, tf.float32) / float(tf.int16.max)}

    if dataset == "librispeech":
        split = "train_clean360"
    else:
        split = "train"

    ds = tfds.load(
        dataset,
        split=split,
        as_supervised=True,
        data_dir="/mnt/storage/cdtdisspotify/tensorflow_datasets/",
    )
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def get_audio_dataset(dataset="crema_d", shuffle_buffer=1000):
    """Read downstream audio task data from TFDS as tf dataset."""

    @tf.function
    def _parse_example(audio, label):
        return {
            "audio": tf.cast(audio, tf.float32) / float(tf.int16.max),
            "label": label,
        }

    (ds_train, ds_test), ds_info = tfds.load(
        dataset,
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir="/mnt/storage/cdtdisspotify/tensorflow_datasets/",
    )

    ds_train = ds_train.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds_train = ds_train.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    ds_test = ds_test.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds_test = ds_test.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return (ds_train, ds_test, ds_info.features["label"].num_classes)


def random_segment(waveform, seg_length=16000):
    """Extract a random segment from a waveform."""
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
    return tf.image.random_crop(padded_waveform, [seg_length])


def extract_log_mel_spectrogram(
    waveform,
    sample_rate=16000,
    frame_length=400,
    frame_step=160,
    fft_length=1024,
    n_mels=64,
    fmin=60.0,
    fmax=7800.0,
):
    """Extract frames of log mel spectrogram from a raw waveform."""

    stfts = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
    )
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    mel_spectrograms = tf.clip_by_value(
        mel_spectrograms, clip_value_min=1e-5, clip_value_max=1e8
    )

    log_mel_spectrograms = tf.math.log(mel_spectrograms)

    return log_mel_spectrograms
