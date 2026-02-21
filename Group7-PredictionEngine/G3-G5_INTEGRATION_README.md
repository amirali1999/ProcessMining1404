# Group 3 -> Group 7 -> Group 5 Integration Guide

This guide explains how Group 3 should send data to Group 7 and how Group 5 should consume prediction results from Group 7.

## 1) Base URL
When running the Group 7 server locally:

- Base URL: http://127.0.0.1:8000
- Group 7 endpoints are under: /prediction/

## 2) Input From Group 3 (XES)
Group 7 trains models from an XES file. Group 3 should POST a preprocessed XES file to the training endpoint.

### Training Endpoint
- URL: POST /prediction/api/train/
- Content-Type: multipart/form-data
- Body fields:
  - xes_file (required): the XES file
  - train_outcome (optional): true/false (default true)
  - train_lstm (optional): true/false (default true)
  - max_case_length (optional): int (default 50)
  - batch_size (optional): int (default 512)
  - epochs_next (optional): int (default 5)
  - epochs_time (optional): int (default 5)

### Example (curl)
```bash
curl -X POST http://127.0.0.1:8000/prediction/api/train/ \
  -F "xes_file=@/path/to/cleaned_log.xes" \
  -F "train_outcome=true" \
  -F "train_lstm=true"
```

### Response (success)
```json
{
  "status": "ok",
  "xes_path": "/abs/path/uploaded_xes/20260220_121500_cleaned_log.xes",
  "models_dir": "/abs/path/trained_models_torch",
  "trained": {
    "outcome": true,
    "lstm_torch": true
  }
}
```

Notes:
- Group 7 uses PyTorch by default for LSTM. TensorFlow remains optional.
- After training, the models are loaded in memory and are ready for prediction.

## 3) Output For Group 5 (Predictions)
Group 5 can call the prediction endpoints to get outcome, remaining time, and next activity.

### Health Check
- URL: GET /prediction/api/health/
- Response indicates whether models are loaded.

### Predict All (recommended)
- URL: POST /prediction/api/predict/all/
- Body options:
  - case_id: string (lookup inside loaded XES log)
  - activities: array of activity names (no lookup)

#### Example (by activities)
```bash
curl -X POST http://127.0.0.1:8000/prediction/api/predict/all/ \
  -H "Content-Type: application/json" \
  -d '{"activities": ["NEW", "CODE OK", "MANUAL"]}'
```

#### Response (shape)
```json
{
  "case_id": "hypothetical",
  "current_activities": ["NEW", "CODE OK", "MANUAL"],
  "prefix_length": 3,
  "suggestion": {
    "next_activity": "BILLED"
  },
  "prediction": {
    "outcome": "Standard Processing",
    "remaining_time": {
      "seconds": 12345.0,
      "minutes": 205.75,
      "hours": 3.43,
      "days": 0.14
    }
  }
}
```

### Predict Outcome Only
- URL: POST /prediction/api/predict/outcome/
- Body: {"case_id": "case_123"}

### Predict Next Activity Only
- URL: POST /prediction/api/predict/next-activity/
- Body: {"case_id": "case_123"}

### Predict Remaining Time Only
- URL: POST /prediction/api/predict/remaining-time/
- Body: {"case_id": "case_123"}

## 4) UI Demo (Optional)
Group 5 can also use the demo UI to test training and predictions:

- URL: http://127.0.0.1:8000/prediction/
- Upload XES and train from the UI.
- Enter activities or case ID to get predictions.

## 5) Data Contract Summary
Inputs (from Group 3):
- XES file with standard columns: case:concept:name, concept:name, time:timestamp

Outputs (to Group 5):
- next_activity (string)
- remaining_time (seconds/minutes/hours/days)
- outcome (string)

If you need an endpoint that accepts a URL/path instead of file upload, ask Group 7 to enable it.
