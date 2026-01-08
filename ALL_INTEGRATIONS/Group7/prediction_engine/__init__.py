"""
Process Mining Prediction Engine

A comprehensive prediction engine for process mining that includes:
- Phase 1: Outcome prediction using classification models
- Phase 2: Next activity and remaining time prediction using LSTM

Author: Group 7 - Process Mining Prediction Engine
"""

__version__ = '1.0.0'
__author__ = 'Group 7'

from .data_preprocessing import XESDataPreprocessor
from .outcome_prediction import OutcomePredictionModel, EnsembleOutcomePredictor
from .lstm_models import NextActivityLSTM, RemainingTimeLSTM, CombinedLSTMPredictor

__all__ = [
    'XESDataPreprocessor',
    'OutcomePredictionModel',
    'EnsembleOutcomePredictor',
    'NextActivityLSTM',
    'RemainingTimeLSTM',
    'CombinedLSTMPredictor',
]
