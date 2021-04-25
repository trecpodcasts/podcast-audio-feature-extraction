# -*- coding: utf-8 -*-

"""Audio features for search reranking."""

import numpy as np
from scipy.special import gamma


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


def gamma_distribution(x, alpha, beta):
    return beta**alpha/gamma(alpha) * x.T**(alpha-1) * np.exp(-beta*x.T)


def gamma_predict(x, alpha, beta, priors):
    p =  gamma_distribution(x, alpha, beta) * priors
    return (p/np.sum(p,axis=0)).T


def gauss_distribution(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)


def gauss_predict(x, alpha, beta, priors):
    p =  gauss_distribution(x, alpha, beta) * priors
    return (p/np.sum(p,axis=0)).T


def proba(x, theta, sigma, priors):
    """Return probability for each category like sklearn predict_proba."""
    p = 1/np.sqrt(2*np.pi*sigma) * np.exp(-0.5 * (x.T - theta.reshape(-1,1))**2/sigma) * priors.reshape(-1,1)
    return (p/np.sum(p,axis=0)).T


#def yamnet_is_funny(yamnet_scores):
#    """Predict if a segment is funny from the YAMNet scores."""
#    a = np.sum(np.max(yamnet_scores[:,1:], axis=1) == yamnet_scores[:,13]) 
#    return proba(
#        np.array([a]),
#        np.array([[0.52702703],[3.18421053]]),
#        np.array([[1.70872902],[35.86080334]]),
#        np.array([0.49333333, 0.50666667]) 
#    )
def yamnet_is_funny(yamnet_scores, threshold=.5):
    """predict if one segment is funny based on YAMNet data"""
    c = np.sum(np.max(yamnet_scores[:,1:], axis=1) == yamnet_scores[:,13]) 
    p = gamma_predict(c,
                      np.array([[0.19338548], [0.33139668]]),
                      np.array([[0.34137539], [0.11298827]]),
                      np.array([[0.47150383], [0.52849617]]))
    return p


def yamnet_is_conversation(yamnet_scores):
    """Predict if a segment is conversation from the YAMNet scores."""
    a = np.sum(np.max(yamnet_scores[:,1:], axis=1) == yamnet_scores[:,2]) 
    return proba(
        np.array([a]),
        np.array([[0.28888889],[0.1047619 ]]),
        np.array([[0.24987654],[0.16997732]]),
        np.array([0.3, 0.7])
    )


#def opensmile_is_debate(opensmile_scores):
#    """Predict if a segment is debate from the openSMILE scores.""" 
#    s0 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 551.696
#    s1 = np.max(opensmile_scores["slopeUV500-1500_sma3nz_amean"].to_numpy()) / 0.015960522
#    return s0 + s1
#def opensmile_is_debate(opensmile_scores):
#    """Predict if a segment is debate from the openSMILE scores.""" 
#    s0 = np.std(opensmile_scores["mfcc4_sma3_stddevNorm"].to_numpy()) / 142.52017
#    s1 = np.max(opensmile_scores["slopeUV500_1500_sma3nz_amean"].to_numpy()) / 0.0155821005
#    return s0 + (12 * s1)
def opensmile_is_debate(opensmile_scores):
    """Predict if a segment is debate from the openSMILE scores."""
    c =  np.mean(opensmile_scores["mfcc4_sma3_stddevNorm"])
    p = gauss_predict(c,
                      np.array([[-1.6886425], [-9.844541 ]]),
                      np.array([[16.348476], [82.52454 ]]),
                      np.array([[0.81867616], [0.18132384]]))
    return p


#def opensmile_is_disapproval(opensmile_scores):
#    """Predict if a segment is disapproval from the openSMILE scores.""" 
#    s0 = np.mean(opensmile_scores["F2frequency_sma3nz_amean"].to_numpy()) / 1579.2253
#    s1 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 551.6959
#    s2 = np.std(opensmile_scores["mfcc2V_sma3nz_amean"].to_numpy()) / 8.204574
#    return s0 + s1 + s2
#def opensmile_is_disapproval(opensmile_scores):
#    """Predict if a segment is disapproval from the openSMILE scores.""" 
#    s0 = np.mean(opensmile_scores["spectralFlux_sma3_stddevNorm"].to_numpy()) / 0.8239882
#    s1 = np.mean(opensmile_scores["F1frequency_sma3nz_amean"].to_numpy()) / 556.0259
#    s2 = np.mean(opensmile_scores["F2frequency_sma3nz_amean"].to_numpy()) / 1586.24
#    return (2 * s0) + s1 + s2
def opensmile_is_disapproval(opensmile_scores):
    """Predict if a segment is disapproval from the openSMILE scores.""" 
    c =  np.mean(opensmile_scores["mfcc4_sma3_stddevNorm"])
    p = gauss_predict(c,
                      np.array([[-1.7017771], [-11.438402 ]]),
                      np.array([[16.199871], [91.853195]]),
                      np.array([[0.84457478], [0.15542522]]))
    return p