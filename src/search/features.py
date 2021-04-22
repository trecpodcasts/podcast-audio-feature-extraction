# -*- coding: utf-8 -*-

"""Audio features for search reranking."""

import numpy as np

def yamnet_freq_feature(yamnet_scores, class_index, threshold_value=0.02):
    """Calculate a simple frequency feature from the YAMNet scores.
    
    Some useful YAMNet class indices:
    13 = Laughter
    132 = Music
    2 = Conversation
    3 = Narration, monologue
    """
    freq_score = (yamnet_scores[:, class_index] > threshold_value).sum()
    return freq_score


def proba(x, theta, sigma, priors):
    """Return probability for each category like sklearn predict_proba."""
    p = 1/np.sqrt(2*np.pi*sigma) * np.exp(-0.5 * (x.T - theta.reshape(-1,1))**2/sigma) * priors.reshape(-1,1)
    return (p/np.sum(p,axis=0)).T


def yamnet_is_funny(yamnet_scores):
    """Predict if a segment is funny from the YAMNet scores."""
    a = np.sum(np.max(yamnet_scores[:,1:], axis=1) == yamnet_scores[:,13]) 
    return proba(
        np.array([a]),
        np.array([[0.52702703],[3.18421053]]),
        np.array([[1.70872902],[35.86080334]]),
        np.array([0.49333333, 0.50666667]) 
    )


def yamnet_is_conversation(yamnet_scores):
    """Predict if a segment is conversation from the YAMNet scores."""
    a = np.sum(np.max(yamnet_scores[:,1:], axis=1) == yamnet_scores[:,2]) 
    return proba(
        np.array([a]),
        np.array([[0.28888889],[0.1047619 ]]),
        np.array([[0.24987654],[0.16997732]]),
        np.array([0.3, 0.7])
    )


def opensmile_is_debate(opensmile_scores):
    """Predict if a segment is debate from the openSMILE scores.""" 
    s0 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 551.696
    s1 = np.max(opensmile_scores["slopeUV500-1500_sma3nz_amean"].to_numpy()) / 0.015960522
    return s0 + s1


def opensmile_is_disapproval(opensmile_scores):
    """Predict if a segment is disapproval from the openSMILE scores.""" 
    s0 = np.mean(opensmile_scores["F2frequency_sma3nz_amean"].to_numpy()) / 1579.2253
    s1 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 551.6959
    s2 = np.std(opensmile_scores["mfcc2V_sma3nz_amean"].to_numpy()) / 8.204574
    return s0 + s1 + s2
