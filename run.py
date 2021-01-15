# -*- coding: utf-8 -*-

"""Main running script."""

import os
import logging

# Get rid of all the annoying tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.CRITICAL)
import tensorflow as tf  # noqa: E402
import pretty_errors  # noqa: F401
import hydra
from omegaconf import DictConfig, OmegaConf


def setup_gpus():
    """Enable memory growth on the GPU's."""
    # Need to setup the GPU's before we import anything else that uses tensorflow
    gpus = tf.config.list_physical_devices("GPU")
    print("Found {} GPUs".format(len(gpus)))
    if tf.config.list_physical_devices("GPU"):
        try:  # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
            print(e)


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function, providing easy access to the configuration.
    Args:
        cfg (DictConfig): configuration dictionary
    """
    setup_gpus()  # Setup the GPUs
    print(
        "Configuration: \n{}".format(OmegaConf.to_yaml(cfg))
    )  # Print the configuration


if __name__ == "__main__":
    main()
