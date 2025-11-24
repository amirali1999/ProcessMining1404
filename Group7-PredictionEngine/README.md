# Process Mining Prediction Engine

## Overview
This project implements a comprehensive Process Mining Prediction Engine using Deep Learning (LSTM) and Machine Learning techniques. It is designed to analyze event logs (XES format) from business processes and predict:
1.  **Next Activity**: What will happen next in the case?
2.  **Remaining Time**: How long until the case is finished?
3.  **Outcome**: What will be the final result of the case?

The system includes a data preprocessing pipeline, model training scripts, and a Django-based REST API with a web demo interface.

## Project Structure

```
project/
├── prediction_engine/          # Main application module
│   ├── api_views.py            # Django API endpoints
│   ├── data_preprocessing.py   # XES parsing and feature extraction
│   ├── lstm_models.py          # LSTM architectures (Next Activity & Time)
│   ├── outcome_prediction.py   # ML models for outcome prediction
│   ├── train_models.py         # Script to train all models
│   ├── test_best_models.py     # Script to verify trained models
│   └── templates/              # HTML templates for the demo
├── process_mining_core/        # Django project configuration
├── trained_models/             # Directory where models are saved
├── HospitalBilling-EventLog_1_all/ # Dataset directory
├── manage.py                   # Django management script
└── requirements.txt            # Python dependencies
```

## Installation

1.  **Prerequisites**: Python 3.10+
2.  **Install Dependencies**:
    ```bash
    pip install -r prediction_engine/requirements.txt
    ```

## Usage

### 1. Training the Models
To train the models from scratch using your XES event log:

```bash
python prediction_engine/train_models.py
```
*Note: Ensure your XES file is in the correct path as defined in the script.*

### 2. Running the Web Demo
Start the Django development server:

```bash
python manage.py runserver
```

Then open your browser and navigate to:
[http://127.0.0.1:8000/prediction/](http://127.0.0.1:8000/prediction/)

### 3. API Endpoints
The system exposes the following API endpoints:

*   `POST /prediction/api/predict/all/`: Get all predictions (Next Activity, Time, Outcome).
    *   **Body**: `{"activities": ["NEW", "CHANGE DIAGN"]}` or `{"case_id": "A"}`

## Technical Details
*   **Next Activity**: Bidirectional LSTM with Focal Loss to handle class imbalance.
*   **Remaining Time**: Regression LSTM predicting log-normalized time.
*   **Outcome**: Ensemble of Random Forest and XGBoost.

## Authors
*   Mohammad Mobin Teymourpour
*   Amirmohammad Hosseini
*   Fatemeh Dehbashi
