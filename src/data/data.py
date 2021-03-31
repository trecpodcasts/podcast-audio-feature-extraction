# -*- coding: utf-8 -*-

"""Provides helper data related functions."""

__all__ = [
    "load_metadata",
    "find_file_paths",
    "find_paths",
    "find_file_paths_features",
    "load_transcript",
    "retrieve_full_transcript",
    "retrieve_timestamped_transcript",
    # "get_podcast_dataset",
    # "get_tfds_dataset",
    # "random_sample",
    # "parse_raw_audio",
    # "extract_log_mel",
    # "extract_functionals",
]

# TODO: Be able to load transcripts as well, maybe want to use huggingface tokeniser on the text first
# TODO: Probably want to be able to load the other metadata we have into a dataset as well

import json
import os
import warnings
import pandas as pd
import numpy as np

import opensmile
import tensorflow as tf

# import tensorflow_io as tfio
# import tensorflow_datasets as tfds


DATA_PATH = "/unix/cdtdisspotify/data/spotify-podcasts-2020/"
TFDS_PATH = "/mnt/storage/cdtdisspotify/tensorflow_datasets/"
DATA_PATH_GPU02 = "/mnt/storage/cdtdisspotify/eGeMAPSv01b/intermediate/"
SAMPLE_RATE = 16000
SMILE = opensmile.Smile(  # Create the functionals extractor here
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def load_metadata():
    """Load the Spotify podcast dataset metadata."""
    return pd.read_csv(DATA_PATH + "metadata.tsv", delimiter="\t")


def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """Return the relative filepath based on the episode metadata."""
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )


def find_paths(metadata, base_folder, file_extension):
    """Finds the filepath based on the dataset structure based on the metadata, the folder where it is stored and the file extension you want.

    Args:
        metadata (df): The metadata of the files you want to create a path for
        base_folder (str): base directory for where data is (to be) stored
        file_extension (str): extension of the file in the path

    Returns:
        paths (list): list of paths (str) for all files in the given metadata
    """

    paths = []
    for i in range(len(metadata)):
        relative_path = relative_file_path(
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i],
        )
        path = os.path.join(base_folder, relative_path + file_extension)
        paths.append(path)
    return paths


def find_file_paths(show_filename_prefix, episode_filename_prefix):
    """Get the transcript and audio paths from show and episode prefix.

    Args:
        show_filename_prefix: As given in metadata-*.tsv
        episode_filename_prefix: As given in metadata-*.tsv

    Returns:
        path to transcript .json file
        path to audio .ogg file

    # TODO also include the sets for summarization-testset
    """
    warning.warn(
        "This function is deprecated, use the find_paths function with the predefined paths in data_paths.py",
        DeprecationWarning,
    )

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


def find_file_paths_features(show_filename_prefix, episode_filename_prefix):
    """Find the feature file on the gpu02 machine

    Args:
        show_filename_prefix: as given in metadata-*.tsv
        episode_filename_prefix: as given in metadata-*.tsv

    Returns:
        path to .pkl feature file on the local storage of the gpu02 machine
    """
    warning.warn(
        "This function is deprecated, use the find_paths function with the predefined paths in data_paths.py",
        DeprecationWarning,
    )
    relative_path = relative_file_path(show_filename_prefix, episode_filename_prefix)

    feature_path = os.path.join(DATA_PATH_GPU02, relative_path + ".pkl")

    return feature_path


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


def retrieve_timestamped_transcript(path):
    """Load the full transcript with timestamps"""
    with open(path, "r") as file:
        transcript = json.load(file)

    starts, ends, words, speakers = [], [], [], []
    for word in transcript["results"][-1]["alternatives"][0]["words"]:
        starts.append(float(word["startTime"].replace("s", "")))
        ends.append(float(word["endTime"].replace("s", "")))
        words.append(word["word"])
        speakers.append(word["speakerTag"])

    starts = np.array(starts, dtype=np.float32)
    ends = np.array(ends, dtype=np.float32)
    words = np.array(words)
    speakers = np.array(speakers, dtype=np.int32)
    return starts, ends, words, speakers


def get_podcast_dataset(
    method,
    sample_length=1,
    shuffle_buffer=1000,
    data_path=None,
    feature="log_mel",
    positive_noise=None,
):
    """Gets the Spotify podcast audio data as a tf.dataset.

    Args:
        method: Type of sample retrieval ['singular', 'sequential', 'random', 'full'].
        sample_length: Optional; Length of sample in seconds.
        shuffle_buffer: Optional; Size of dataset shuffle buffer.
        data_path: Optional; Path to data directory to use instead.
        feature: Optional; Feature to extract from the raw waveforms.
        positive_noise: Optional; Noise scaling value to use for positive samples.
    """

    @tf.function
    def _parse_singular(file_path):
        """Parse a file to a single sample."""
        lazy = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
        anchor = random_sample(lazy, lazy.rate, sample_length, input_type="lazy")
        anchor = parse_raw_audio(anchor, lazy.rate)
        return {"anchor": anchor}

    @tf.function
    def _parse_sequential(file_path):
        """Parse a file to a sequential pair of samples."""
        lazy = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
        sample = random_sample(lazy, lazy.rate, sample_length * 2, input_type="lazy")
        sample = parse_raw_audio(sample, lazy.rate)
        anchor, positive = tf.split(sample, 2, axis=0)
        if positive_noise is not None:
            positive = positive + (
                positive_noise * tf.random.normal(tf.shape(positive))
            )
        return {"anchor": anchor, "positive": positive}

    @tf.function
    def _parse_random(file_path):
        """Parse a file to a random pair of samples."""
        lazy = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
        anchor = random_sample(lazy, lazy.rate, sample_length, input_type="lazy")
        anchor = parse_raw_audio(anchor, lazy.rate)
        positive = random_sample(lazy, lazy.rate, sample_length, input_type="lazy")
        positive = parse_raw_audio(positive, lazy.rate)
        if positive_noise is not None:
            positive = positive + (
                positive_noise * tf.random.normal(tf.shape(positive))
            )
        return {"anchor": anchor, "positive": positive}

    @tf.function
    def _parse_full(file_path):
        """Parse a full podcast file to a sequence of frames of sample_length."""
        lazy = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
        anchor = parse_raw_audio(lazy[:], lazy.rate, frame_length=sample_length)
        return {"anchor": anchor}

    @tf.function
    def _extract_features(example):
        """Extract the required features and unpack the example dictionary."""
        if feature == "log_mel":
            if len(list(example.keys())) == 1:
                anchor = extract_log_mel(example["anchor"])
                return anchor
            else:
                anchor = extract_log_mel(example["anchor"])
                positive = extract_log_mel(example["positive"])
                return anchor, positive
        elif feature == "functionals":
            if len(list(example.keys())) == 1:
                anchor = extract_functionals(example["anchor"])
                return anchor
            else:
                anchor = extract_functionals(example["anchor"])
                positive = extract_functionals(example["positive"])
                return anchor, positive
        else:
            if len(list(example.keys())) == 1:
                return example["anchor"]
            else:
                return example["anchor"], example["positive"]

    # Get the audio file data paths
    if data_path:
        paths = [os.path.join(data_path, path) for path in os.listdir(data_path)]
    else:
        metadata = load_metadata()
        episode_pre = metadata["episode_filename_prefix"]
        show_pre = metadata["show_filename_prefix"]
        paths = [find_file_paths(sp, ep)[1] for sp, ep in zip(show_pre, episode_pre)]

    # Create the dataset, shuffle, and load data
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # Parse the audio files into singular, sequential or random samples
    # We can not parallelise this process or everything breaks, hence num_parallel_calls=1
    if method == "singular":
        ds = ds.map(_parse_singular, num_parallel_calls=1)
    elif method == "sequential":
        ds = ds.map(_parse_sequential, num_parallel_calls=1)
    elif method == "random":
        ds = ds.map(_parse_random, num_parallel_calls=1)
    elif method == "full":
        ds = ds.map(_parse_full, num_parallel_calls=1)
    else:
        raise ValueError(
            "Method arguement needs to be ['singular', 'sequential', 'random']"
        )

    # Extract audio features from the raw audio if required and return dataset
    # We can not parallelise for functionals or everything breaks, hence num_parallel_calls=1
    if feature == "functionals":
        ds = ds.map(_extract_features, num_parallel_calls=1)
    else:
        ds = ds.map(_extract_features, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_tfds_dataset(
    name,
    sr,
    method,
    sample_length=1,
    shuffle_buffer=1000,
    feature="log_mel",
    split="train",
    positive_noise=None,
    silence_epsilon=0.002,
):
    """Gets a TFDS audio dataset as a tf.dataset.

    Args:
        name: Name of the tf dataset to use
        sr: Sample rate of the tf dataset
        method: Type of sample retrieval ['singular', 'sequential', 'random', 'full'].
        sample_length: Optional; Length of sample in seconds.
        shuffle_buffer: Optional; Size of dataset shuffle buffer.
        feature: Optional; Feature to extract from the raw waveforms.
        split: Optional; Which tf dataset split to return.
        positive_noise: Optional; Noise scaling value to use for positive samples.
        silence_epsilon: Optional; Max value to be considered as noise when trimming silence.
    """

    @tf.function
    def _parse_singular(y, label):
        """Parse the audio to a single sample."""
        y = parse_raw_audio(y, sr)
        if silence_epsilon:
            y = trim_silence(y, silence_epsilon, SAMPLE_RATE * sample_length)
        anchor = random_sample(y, SAMPLE_RATE, sample_length, input_type="tfds")
        return {"anchor": anchor, "label": label}

    @tf.function
    def _parse_sequential(y, label):
        """Parse the audio to a sequential pair of samples."""
        y = parse_raw_audio(y, sr)
        if silence_epsilon:
            y = trim_silence(y, silence_epsilon, SAMPLE_RATE * sample_length)
        sample = random_sample(y, SAMPLE_RATE, sample_length * 2, input_type="tfds")
        anchor, positive = tf.split(sample, 2, axis=0)
        if positive_noise is not None:
            positive = positive + (
                positive_noise * tf.random.normal(tf.shape(positive))
            )
        return {"anchor": anchor, "positive": positive, "label": label}

    @tf.function
    def _parse_random(y, label):
        """Parse the audio to a random pair of samples."""
        y = parse_raw_audio(y, sr)
        if silence_epsilon:
            y = trim_silence(y, silence_epsilon, SAMPLE_RATE * sample_length)
        anchor = random_sample(y, SAMPLE_RATE, sample_length, input_type="tfds")
        positive = random_sample(y, SAMPLE_RATE, sample_length, input_type="tfds")
        if positive_noise is not None:
            positive = positive + (
                positive_noise * tf.random.normal(tf.shape(positive))
            )
        return {"anchor": anchor, "positive": positive, "label": label}

    @tf.function
    def _parse_full(y, label):
        """Parse the audio to a sequence of frames of sample_length."""
        y = parse_raw_audio(y, sr)
        if silence_epsilon:
            y = trim_silence(y, silence_epsilon, SAMPLE_RATE * sample_length)
        anchor = parse_raw_audio(y, sr, frame_length=sample_length)
        return {"anchor": anchor, "label": label}

    @tf.function
    def _extract_features(example):
        """Extract the required features and unpack the example dictionary."""
        if feature == "log_mel":
            if len(list(example.keys())) == 2:
                anchor = extract_log_mel(example["anchor"])
                return anchor, example["label"]
            else:
                anchor = extract_log_mel(example["anchor"])
                positive = extract_log_mel(example["positive"])
                return anchor, positive
        elif feature == "functionals":
            if len(list(example.keys())) == 2:
                anchor = extract_functionals(example["anchor"])
                return anchor, example["label"]
            else:
                anchor = extract_functionals(example["anchor"])
                positive = extract_functionals(example["positive"])
                return anchor, positive
        else:
            if len(list(example.keys())) == 2:
                return example["anchor"], example["label"]
            else:
                return example["anchor"], example["positive"]

    ds, ds_info = tfds.load(
        name,
        split=[split],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=TFDS_PATH,
    )
    ds = ds[0].shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # Parse the audio files into singular, sequential or random samples
    if method == "singular":
        ds = ds.map(_parse_singular, num_parallel_calls=tf.data.AUTOTUNE)
    elif method == "sequential":
        ds = ds.map(_parse_sequential, num_parallel_calls=tf.data.AUTOTUNE)
    elif method == "random":
        ds = ds.map(_parse_random, num_parallel_calls=tf.data.AUTOTUNE)
    elif method == "full":
        ds = ds.map(_parse_full, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise ValueError(
            "Method arguement needs to be ['singular', 'sequential', 'random']"
        )

    # Extract audio features from the raw audio if required and return dataset
    # We can not parallelise for functionals or everything breaks, hence num_parallel_calls=1
    if feature == "functionals":
        ds = ds.map(_extract_features, num_parallel_calls=1)
    else:
        ds = ds.map(_extract_features, num_parallel_calls=tf.data.AUTOTUNE)
    return ds, ds_info.features["label"].num_classes


@tf.function
def trim_silence(y, epsilon, min_length):
    """Trim silence from either end of the audio data.

    Args:
        y: Raw input audio data.
        epsilon: The max value to be considered as noise.
        sample_length: Minimum length of sample to return.
    """
    trim_boundary = tfio.experimental.audio.trim(y, axis=0, epsilon=epsilon)
    if (trim_boundary[1] - trim_boundary[0]) < min_length:
        return y
    else:
        return y[trim_boundary[0] : trim_boundary[1]]


@tf.function
def random_sample(y, sr, sample_length, input_type="lazy"):
    """Get a random sample from the audio data.

    Args:
        y: Raw input audio data.
        sr: Sample rate of the input audio data.
        sample_length: Sample length of audio to extract in seconds.
    """
    sample_length = tf.cast(sr * sample_length, dtype=tf.int64)
    if input_type == "lazy":
        rand_max = tf.cast(y.shape[0], dtype=tf.int64) - sample_length
    else:
        rand_max = tf.cast(tf.shape(y)[0], dtype=tf.int64) - sample_length

    if rand_max > 0:  # The audio is long enough to pick a random sample
        sample_start = tf.random.uniform(shape=[], dtype=tf.int64, maxval=rand_max)
        return y[sample_start : (sample_start + sample_length)]
    else:  # The audio is shorter than the sample length, return padded version
        if input_type == "lazy":
            y = y[:]
        else:
            y = tf.expand_dims(y, axis=1)
        return tf.pad(
            y, [[0, tf.math.abs(rand_max)], [0, 0]], "CONSTANT", constant_values=0
        )


@tf.function
def parse_raw_audio(y, sr, frame_length=None):
    """Parse the raw audio waveform into a standardised format.

    Args:
        y: Raw input audio data.
        sr: Sample rate of the input audio data.
        frame_length: Optional; Length of frames in seconds to split sample into.
    """
    # Cast to float32 and scale to [-1,1]
    y = tf.cast(y, tf.float32) / float(tf.int16.max)

    # Convert to mono if it is stereo
    if len(tf.shape(y)) != 1:
        y = tf.math.reduce_mean(y, axis=1)

    # Resample to 16000 Hz
    sr = tf.cast(sr, dtype=tf.int64)
    y = tfio.audio.resample(y, sr, SAMPLE_RATE)

    # Split into 'frame_length' long frames if required
    # TODO: For evaluation, maybe we want to average over overlapping frames!
    if frame_length:
        y = tf.signal.frame(
            y,
            frame_length=tf.cast(SAMPLE_RATE * frame_length, tf.int32),
            frame_step=tf.cast(SAMPLE_RATE * frame_length, tf.int32),
            pad_end=False,
        )

    # Apply L2 norm and return parsed audio waveform
    y = tf.math.l2_normalize(y, axis=-1, epsilon=1e-9)
    return y


@tf.function
def extract_log_mel(
    y,
    sr=SAMPLE_RATE,
    frame_length=int(SAMPLE_RATE * 0.025),  # 25 ms
    frame_step=int(SAMPLE_RATE * 0.01),  # 10 ms
    fft_length=1024,
    n_mels=64,
    fmin=60.0,
    fmax=7800.0,
):
    """Extract frames of log mel spectrogram from an audio sample.

    Args:
        y: Input audio data.
        sr: Optional; Sample rate of the input data.
        frame_length: Optional; Length of stft frames.
        frame_step: Optional; Step between stft frames.
        fft_length: Optional; Size of tje FFT to apply.
        n_mels: Optional; Number of mel bands to use.
        fmin: Optional; Lower edge of mel bands.
        fmin: Optional; Upper edge of mel bands.
    """
    stfts = tf.signal.stft(
        y,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
    )
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = tf.cast((fft_length / 2) + 1, dtype=tf.int64)
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sr,
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

    return log_mel_spectrograms[Ellipsis, tf.newaxis]


def extract_functionals(y, sr=SAMPLE_RATE):
    """Extract eGeMAPSv02 functionals from audio sample.

    Args:
        y: Input audio data.
        sr: Optional; Sample rate of the input data.
    """

    # Need to wrap the opensmile stuff in a tf.py_function to work
    def _extract(y, sr):
        funcs = SMILE.process_signal(y.numpy(), sr.numpy())
        funcs = tf.convert_to_tensor(funcs, dtype=tf.float32)
        funcs = tf.reshape(funcs, [88])
        return funcs

    funcs = tf.py_function(_extract, (y, sr), tf.float32)
    return funcs
