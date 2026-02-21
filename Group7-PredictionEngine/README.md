# Process Mining Prediction Engine

## Overview
This project implements a Process Mining Prediction Engine using Deep Learning (LSTM) and classic ML. It analyzes XES event logs and predicts:
1. **Next Activity**: What happens next?
2. **Remaining Time**: How long until completion?
3. **Outcome**: Final case result.

The stack uses PyTorch as the default backend for sequence models, with TensorFlow/Keras kept optional for compatibility. It ships with a Django REST API and a dark-mode demo UI that supports XES upload and model training.

## Project Structure

```
project/
├── prediction_engine/           # Main application module
│   ├── api_views.py             # Django API endpoints
│   ├── data_preprocessing.py    # XES parsing and feature extraction
│   ├── torch_models.py          # Torch LSTM models (default backend)
│   ├── lstm_models.py           # TensorFlow LSTM models (optional)
│   ├── outcome_prediction.py    # Classic ML models for outcome prediction
│   ├── train_models.py          # Script to train all models
│   ├── test_best_models.py      # Script to verify trained models
│   └── templates/               # HTML templates for the demo
├── process_mining_core/         # Django project configuration
├── trained_models_torch/        # Torch models and preprocessor
├── trained_models/              # TensorFlow models (optional)
├── dataset/                     # Sample dataset
├── manage.py                    # Django management script
└── requirements.txt             # Python dependencies
```

## Installation

1. **Prerequisites**: Python 3.10+
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Models (API)
You can train by uploading a XES file directly from the demo UI or via API.

**API**:
`POST /prediction/api/train/` (multipart form)
- `xes_file` (required)
- `train_outcome` (true/false)
- `train_lstm` (true/false)
- `max_case_length`, `batch_size`, `epochs_next`, `epochs_time` (optional)

### 2. Running the Web Demo
```bash
python manage.py runserver
```

Open:
http://127.0.0.1:8000/prediction/

### 3. API Endpoints
* `POST /prediction/api/predict/all/`: All predictions.
    * Body: `{"activities": ["NEW", "CHANGE DIAGN"]}` or `{"case_id": "A"}`
* `POST /prediction/api/predict/outcome/`
* `POST /prediction/api/predict/next-activity/`
* `POST /prediction/api/predict/remaining-time/`
* `GET /prediction/api/health/`

## Technical Details
* **Next Activity**: Torch LSTM classifier (TF optional).
* **Remaining Time**: Torch LSTM regression on log-normalized time.
* **Outcome**: Ensemble of Random Forest and Decision Tree.

## Authors
* Mohammad Mobin Teymourpour
* Amirmohammad Hosseini
* Fatemeh Dehbashi
