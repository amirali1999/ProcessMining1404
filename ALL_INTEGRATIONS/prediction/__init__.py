"""
Prediction App - Group 7 Integration
=====================================
Process prediction using pre-trained models (NO training in web app).

Core components:
- data_preprocessing.py: XES preprocessing, feature extraction (from Group 7)
- outcome_prediction.py: Ensemble outcome prediction (from Group 7)
- lstm_models.py: LSTM for next activity & remaining time (from Group 7)
- services.py: Clean service layer for inference (OUR integration code)
- views.py: DRF API endpoints (OUR integration code)
"""

import os
from django.conf import settings

# Path to trained models directory
TRAINED_MODELS_DIR = os.path.join(settings.BASE_DIR, "prediction", "trained_models")
