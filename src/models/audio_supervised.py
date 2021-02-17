# -*- coding: utf-8 -*-

"""Supervised model for fine-tuning, random encoder and from scratch training."""

import logging
import functools
import os

import tensorflow as tf

import src.data as data
import src.models as models

log = logging.getLogger(__name__)


class SupervisedModel:
    """Provides functionality for self-supervised source separation model."""

    def __init__(self, strategy, cfg):
        """Initializes a supervised model object."""
        # First we create the dataset
        self.train_ds, num_classes = data.get_tfds_dataset(
            cfg.data_name,
            cfg.data_sr,
            "singular",
            sample_length=cfg.data_sample_length,
            shuffle_buffer=cfg.data_shuffle_buffer,
            feature=cfg.data_feature,
            split=cfg.data_train_split,
        )
        self.train_ds = self.train_ds.batch(cfg.data_batch_size, drop_remainder=True)
        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)

        self.test_ds, num_classes = data.get_tfds_dataset(
            cfg.data_name,
            cfg.data_sr,
            "full",
            sample_length=cfg.data_sample_length,
            shuffle_buffer=cfg.data_shuffle_buffer,
            feature=cfg.data_feature,
            split=cfg.data_test_split,
        )
        self.test_ds = self.test_ds.batch(1, drop_remainder=True)
        self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

        if cfg.load_pretrained:
            ckpt_path = os.path.join(
                cfg.pretrained_path, "./model", cfg.pretrained_ckpt
            )
            pretrained_model = models.get_contrastive_network(
                embedding_dim=cfg.contrastive_embedding_dim,
                temperature=cfg.contrastive_temperature,
                pooling_type=cfg.contrastive_pooling_type,
                similarity_type=cfg.contrastive_similarity_type,
            )
            pretrained_model.compile(
                optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
            pretrained_model.load_weights(ckpt_path).expect_partial()
            encoder = pretrained_model.embedding_model.get_layer("encoder")
        else:
            encoder = models.get_efficient_net_encoder(
                pooling=cfg.contrastive_pooling_type
            )

        inputs = tf.keras.layers.Input(shape=(None, 64, 1))
        x = encoder(inputs)
        outputs = tf.keras.layers.Dense(num_classes, activation=None)(x)
        self.model = tf.keras.Model(inputs, outputs)
        if cfg.freeze_encoder:
            self.model.get_layer("encoder").trainable = False
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        self.model.summary()

        self._epochs = cfg.epochs
        self._learning_rate = cfg.learning_rate

    def train_eval(self):
        """Trains and evaluates a downstream model in any of the below mentioned modes."""
        backandrestore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir="./model/backup"
        )

        self.model.fit(
            self.train_ds,
            epochs=self._epochs,
            verbose=1,
            callbacks=[backandrestore_callback],
        )

        time_distributed_input = tf.keras.layers.Input(shape=(None, None, 64, 1))
        x = tf.keras.layers.TimeDistributed(self.model)(time_distributed_input)
        time_averaged_output = tf.reduce_mean(x, axis=1)
        time_distributed_model = tf.keras.Model(
            time_distributed_input, time_averaged_output
        )
        time_distributed_model.compile(
            optimizer=tf.keras.optimizers.Adam(self._learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        test_loss, test_acc = time_distributed_model.evaluate(self.test_ds, verbose=2)

        log.info("Final test loss: %f", test_loss)
        log.info("Final test accuracy: %f", test_acc)
