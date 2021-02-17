# -*- coding: utf-8 -*-

"""Self-supervised model for the contrastive learning task."""


import tensorflow as tf

import src.data as data
import src.models as models


class ContrastiveModel:
    """Provides functionality for self-supervised constrastive learning model."""

    def __init__(self, strategy, cfg):
        """Initializes a contrastive model object."""
        # First we create the dataset
        if cfg.data_name == "podcasts":
            self.ds = data.get_podcast_dataset(
                cfg.data_method,
                sample_length=cfg.data_sample_length,
                shuffle_buffer=cfg.data_shuffle_buffer,
                data_path=cfg.data_path,
                feature=cfg.data_feature,
                positive_noise=cfg.data_positive_noise,
            ).repeat()
        else:
            self.ds, _ = data.get_tfds_dataset(
                cfg.data_name,
                cfg.data_sr,
                cfg.data_method,
                sample_length=cfg.data_sample_length,
                shuffle_buffer=cfg.data_shuffle_buffer,
                feature=cfg.data_feature,
                split=cfg.data_split,
                positive_noise=cfg.data_positive_noise,
            )
        self.ds = self.ds.batch(cfg.data_batch_size, drop_remainder=True)
        self.ds = self.ds.prefetch(tf.data.AUTOTUNE)
        self.ds = strategy.experimental_distribute_dataset(
            self.ds
        )  # TODO: Check what this does

        # Now we create the contrastive model
        with strategy.scope():
            self.model = models.get_contrastive_network(
                embedding_dim=cfg.model_embedding_dim,
                temperature=cfg.model_temperature,
                pooling_type=cfg.model_pooling_type,
                similarity_type=cfg.model_similarity_type,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(cfg.model_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

        self._epochs = cfg.train_epochs
        self._steps_per_epoch = cfg.train_steps_per_epoch

    def train(self):
        """Trains a self-supervised model for contrastive learning."""
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./model/ckpt_{epoch}", save_weights_only=True, monitor="loss"
        )
        backandrestore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir="./model/backup"
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log")

        self.model.fit(
            self.ds,
            epochs=self._epochs,
            steps_per_epoch=self._steps_per_epoch,
            verbose=1,
            callbacks=[
                model_checkpoint_callback,
                backandrestore_callback,
                tensorboard_callback,
            ],
        )
