# -*- coding: utf-8 -*-

"""Generic utilities."""

import subprocess as sp

import tensorflow as tf


def gpu_setup(level=7000):
    """Set up memory growth on all available GPUs."""

    def _output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    free = [True if val > level else False for val in values]
    gpus = tf.config.list_physical_devices("GPU")
    to_use = []
    for free, gpu in zip(free, gpus):
        if free:
            tf.config.experimental.set_memory_growth(gpu, True)
            to_use.append(gpu)
    try:
        tf.config.experimental.set_visible_devices(to_use, "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "actual GPUs,", len(logical_gpus), "in use.")
    except RuntimeError as e:
        print(e)
