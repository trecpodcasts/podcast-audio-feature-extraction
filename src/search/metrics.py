# -*- coding: utf-8 -*-

"""Audio metrics for search reranking."""

import numpy as np


def yamnet_freq_feature(yamnet_scores, class_index, threshold_value=0.02):
    """Calculate a simple frequency feature from the YAMNet scores.

    Some useful YAMNet class indices:
    13 = Laughter
    132 = Music
    2 = Conversation
    3 = Narration, monologue
    """
    try:
        freq_score = (yamnet_scores[:, class_index] > threshold_value).sum()
    except Exception:
        freq_score = 0.0
    return freq_score


def yamnet_is_funny(yamnet_scores, threshold=0.5):
    """Predict if one segment is funny based on YAMNet data."""
    try:
        p = np.sum(np.max(yamnet_scores[:, 1:], axis=1) == yamnet_scores[:, 13])
    except Exception:
        p = 0.0
    return p


def opensmile_is_debate(opensmile_scores):
    """Predict if a segment is debate from the openSMILE scores."""
    try:
        s0 = np.std(opensmile_scores["mfcc4_sma3_stddevNorm"].to_numpy()) / 142.52017
        s1 = (
            np.max(opensmile_scores["slopeUV500-1500_sma3nz_amean"].to_numpy())
            / 0.0155821005
        )
        return s0 + (12 * s1)
    except Exception:
        return 0.0


def opensmile_is_disapproval(opensmile_scores):
    """Predict if a segment is disapproval from the openSMILE scores."""
    try:
        s0 = (
            np.mean(opensmile_scores["spectralFlux_sma3_stddevNorm"].to_numpy())
            / 0.8239882
        )
        s1 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 556.0259
        s2 = np.mean(opensmile_scores["F2frequency_sma3nz_amean"].to_numpy()) / 1586.24
        return (2 * s0) + s1 + s2
    except Exception:
        return 0.0
