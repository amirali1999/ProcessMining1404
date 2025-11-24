# Bucket-Based Predictive Process Monitoring Engine

**گروه ۷: موتور پیش‌بینی فرآیند (Predictive Process Monitoring)**

## 📋 Overview

This implementation follows the **Prefix-Length Bucketing** strategy for predictive process monitoring:

- **Phase 1**: Outcome prediction using classic ML (Decision Tree, Logistic Regression, Random Forest)
- **Phase 2**: Next activity and remaining time prediction using bucket-specific LSTM/GRU models

### Key Innovation: Bucket-Based Architecture

Instead of one model handling all sequence lengths, we create **separate models for each prefix length**:
- Bucket "1": Model for sequences of length 1
- Bucket "2": Model for sequences of length 2
- ...
- Bucket "10+": Model for sequences of length ≥ 10

This approach:
- ✅ Handles variable-length sequences naturally
- ✅ Each model specializes in specific process stages
- ✅ No padding needed within buckets
- ✅ Better accuracy than single-model approaches

## 🏗️ Architecture

```
bucket_prediction_engine/
├── data_loader.py          # گام ۰-۱: XES loading, prefix generation, bucketing
├── outcome_models.py       # فاز ۱: Classic ML for outcome prediction
├── bucket_models.py        # فاز ۲: Bucket-specific LSTM/GRU models
├── train_models.py         # Main training script
├── api_views.py            # گام ۶: Django REST API
├── urls.py                 # API URL routing
└── trained_models/         # Saved models
    ├── outcome/            # Phase 1 models
    └── buckets/            # Phase 2 bucket models
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

**Train both phases:**
```bash
python train_models.py ../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes
```

**Train only Phase 1 (outcome prediction):**
```bash
python train_models.py ../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes --phase1-only
```

**Train only Phase 2 (bucket models):**
```bash
python train_models.py ../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes --phase2-only
```

**Custom parameters:**
```bash
python train_models.py <xes_file> \
  --output-dir trained_models \
  --max-bucket 10 \
  --epochs 20
```

### 3. Use the API

The API provides endpoints for predictions:

**Predict next activity:**
```bash
curl -X POST http://localhost:8000/api/predict/next-activity/ \
  -H "Content-Type: application/json" \
  -d '{"case_id": "H12345"}'
```

**Predict remaining time:**
```bash
curl -X POST http://localhost:8000/api/predict/remaining-time/ \
  -H "Content-Type: application/json" \
  -d '{"case_id": "H12345"}'
```

**Predict outcome:**
```bash
curl -X POST http://localhost:8000/api/predict/outcome/ \
  -H "Content-Type: application/json" \
  -d '{"case_id": "H12345"}'
```

**Get all predictions:**
```bash
curl -X POST http://localhost:8000/api/predict/all/ \
  -H "Content-Type: application/json" \
  -d '{"case_id": "H12345"}'
```

**Use custom prefix (without case_id):**
```bash
curl -X POST http://localhost:8000/api/predict/next-activity/ \
  -H "Content-Type: application/json" \
  -d '{"prefix": ["Registration", "Triage", "XRay"]}'
```

## 📊 Implementation Details

### گام ۰: Load and Prepare Traces

```python
from data_loader import EventLogProcessor

processor = EventLogProcessor('event_log.xes')
processor.build_traces()
```

### گام ۱: Generate Prefixes

```python
prefix_samples = processor.generate_prefixes()
# From trace [A, B, C, D] generates:
# [A] → next: B
# [A, B] → next: C
# [A, B, C] → next: D
```

### گام ۲: Bucket by Length

```python
from data_loader import PrefixBucketer

bucketer = PrefixBucketer(max_bucket=10)
buckets = bucketer.bucket_prefixes(prefix_samples)
```

### گام ۳: Encode Activities

```python
from data_loader import ActivityEncoder

encoder = ActivityEncoder()
encoder.fit(df)
encoded = encoder.encode_prefix(['Registration', 'Triage'])
```

### فاز ۱: Train Outcome Predictor

```python
from outcome_models import OutcomePredictor

predictor = OutcomePredictor(encoder)
X, y = predictor.prepare_data(prefix_samples, max_prefix_length=5)
predictor.train(X_train, y_train, X_val, y_val)
```

### فاز ۲: Train Bucket Models

```python
from bucket_models import BucketEnsemble

ensemble = BucketEnsemble(vocab_size=encoder.vocab_size, max_bucket=10)
bucket_data = ensemble.prepare_bucket_data(buckets, encoder)
ensemble.train_all_buckets(bucket_data, epochs=10)
```

### گام ۸: Evaluation Metrics

**Outcome Prediction:**
- Accuracy
- F1-Score (weighted)
- Confusion Matrix

**Next Activity:**
- Accuracy
- Top-3 Accuracy
- Top-5 Accuracy
- Per-class accuracy

**Remaining Time:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE

## 🔗 Integration with Other Groups

### Group 3 (Data Preprocessing)
- **Input**: Cleaned event log with columns: `case_id`, `activity`, `timestamp`, `outcome`
- **Format**: XES or DataFrame
- **Requirement**: Events sorted by case_id and timestamp

### Group 5 (Visualization)
- **Output**: REST API endpoints for predictions
- **Format**: JSON responses with predictions and confidence scores
- **Features**: Real-time predictions, top-k candidates, remaining time estimates

## 📈 Performance Optimization

1. **Class Weighting**: Handles imbalanced outcome distributions
2. **Bucket Specialization**: Each model learns specific process stages
3. **Top-k Accuracy**: Provides alternative predictions for uncertainty
4. **Early Stopping**: Prevents overfitting with validation monitoring

## 🎯 Training Tips

- **Small dataset**: Use fewer buckets (e.g., 1-5, then 6+)
- **Large dataset**: More granular buckets (1-10, then 11+)
- **Imbalanced classes**: Automatic class weighting applied
- **Quick testing**: Use `--epochs 5` for faster training

## 📝 Example Output

```json
{
  "case_id": "H12345",
  "current_prefix": ["Registration", "Triage", "XRay"],
  "predictions": {
    "next_activity": {
      "activity": "Diagnosis",
      "top_k_candidates": [
        {"activity": "Diagnosis", "prob": 0.62},
        {"activity": "LabTest", "prob": 0.21},
        {"activity": "CTScan", "prob": 0.09}
      ]
    },
    "remaining_time": {
      "hours": 12.5,
      "bucket_id": "3"
    },
    "outcome": {
      "prediction": "Discharged",
      "confidence": 0.85
    }
  }
}
```

## 🛠️ Troubleshooting

**Models not loading:**
- Ensure `trained_models/` directory exists
- Check that all models were saved successfully

**API errors:**
- Verify Django settings include `rest_framework`
- Load event log in `api_views.py` with `load_event_log()`

**Low accuracy:**
- Increase training epochs
- Try different max_bucket values
- Check data quality and class balance

## 📚 References

- **Process Mining**: pm4py library documentation
- **Bucketing Strategy**: Tax et al. (2017) - "Predictive Business Process Monitoring"
- **LSTM for PM**: Evermann et al. (2017) - "Predicting process behaviour using deep learning"

---

**گروه ۷ - Process Mining Project 2025**
