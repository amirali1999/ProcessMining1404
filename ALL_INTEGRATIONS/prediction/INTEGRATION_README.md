# Group 7 - Prediction Engine Integration

## ✅ Integration Complete

The Prediction Engine has been successfully integrated into the main Process Mining web application.

## 📁 Files Created/Modified

### New Files Created:
1. **prediction/__init__.py** - App configuration with TRAINED_MODELS_DIR
2. **prediction/services.py** - Core service layer (inference only, NO training)
3. **prediction/serializers.py** - DRF input validation
4. **prediction/views.py** - DRF ViewSet + web view
5. **prediction/urls.py** - URL configuration (API + web)
6. **templates/prediction/prediction.html** - UI page for predictions

### Files Modified:
1. **config/settings.py** - Added 'prediction.apps.PredictionConfig' to INSTALLED_APPS
2. **config/urls.py** - Added prediction API and web URLs

### Copied from Group 7:
- **prediction/data_preprocessing.py** (from Group7/prediction_engine/)
- **prediction/outcome_prediction.py** (from Group7/prediction_engine/)
- **prediction/lstm_models.py** (from Group7/prediction_engine/)
- **prediction/trained_models/** (all pre-trained models)

## 🎯 What We Integrated

### Core Functionality:
- ✅ **Outcome Prediction**: Ensemble classifier (DT, LR, RF) predicts case outcomes
- ✅ **Next Activity Prediction**: LSTM predicts most likely next activity with top-5 probabilities
- ✅ **Remaining Time Prediction**: LSTM estimates time until case completion (seconds/hours/days)
- ✅ **All-in-One Prediction**: Combined endpoint for all three predictions

### Architecture:
```
User Request
    ↓
DRF API Endpoint (/api/event-logs/{id}/prediction/all/)
    ↓
Service Layer (prediction/services.py)
    ↓
├─ load_prediction_models() → Lazy load + cache
├─ get_log_for_prediction() → Use existing EventLog helpers
└─ predict_outcome/next_activity/remaining_time()
    ↓
Group 7 Models (data_preprocessing, outcome_prediction, lstm_models)
    ↓
Return JSON Response
```

## 🔌 API Endpoints

### Main Endpoint:
- **POST** `/api/event-logs/{id}/prediction/all/`
  - Body: `{"source": "default", "case_id": "case_123"}`
  - Returns: Outcome + Next Activity + Remaining Time

### Individual Endpoints:
- **POST** `/api/event-logs/{id}/prediction/outcome/`
- **POST** `/api/event-logs/{id}/prediction/next-activity/`
- **POST** `/api/event-logs/{id}/prediction/remaining-time/`

### Health Check:
- **GET** `/api/prediction/health/`
  - Returns model loading status

### Web UI:
- **GET** `/prediction/`
  - Interactive prediction page

## 📊 Models Used

### Preprocessor:
- **Location**: `prediction/trained_models/preprocessor.pkl`
- **Purpose**: Label encoders, scalers for feature transformation

### Outcome Model:
- **Location**: `prediction/trained_models/ensemble/`
- **Models**: Decision Tree, Logistic Regression, Random Forest
- **Method**: Voting classifier ensemble

### LSTM Models:
- **Location**: `prediction/trained_models/lstm/`
- **Models**: 
  - `best_next_activity_model.keras` - Next activity prediction
  - `best_remaining_time_model.keras` - Remaining time prediction
- **Architecture**: LSTM with attention mechanism

## 🔄 Data Flow

1. **Input**: User selects EventLog and Case ID
2. **Fetch**: `get_log_for_prediction()` uses existing helpers:
   - `get_default_event_log_df()` for "default" source
   - `get_event_log_dataframe()` for "raw"/"cleaned"
3. **Build Prefix**: Extract activity sequence for selected case
4. **Preprocess**: Apply Group 7's encoders/scalers
5. **Predict**: Run through trained models
6. **Return**: JSON with predictions

## 🚀 Key Features

### Lazy Model Loading:
- Models loaded once on first prediction request
- Cached globally to avoid reloading
- Prints status messages for debugging

### Integration with Main App:
- Uses existing EventLog model and helpers
- NO direct XES file reading
- Respects user permissions (IsAuthenticated)
- Follows same API patterns as Groups 3-6

### No Training Code:
- ALL training code excluded from integration
- Only inference on pre-trained models
- No `train_models.py`, `test_best_models.py`, etc.

## 📖 Usage Example

### From JavaScript (UI):
```javascript
const response = await fetch(`/api/event-logs/1/prediction/all/`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
    },
    body: JSON.stringify({
        source: 'default',
        case_id: 'case_123'
    })
});

const result = await response.json();
// result.outcome.predicted_outcome
// result.next_activity.predicted_next_activity
// result.remaining_time.predicted_remaining_time_hours
```

### From Python (services):
```python
from prediction import services

# All predictions
result = services.predict_all(
    event_log_id=1,
    source="default",
    case_id="case_123"
)

# Individual predictions
outcome = services.predict_outcome(event_log_id=1, case_id="case_123")
next_act = services.predict_next_activity(event_log_id=1, case_id="case_123")
time = services.predict_remaining_time(event_log_id=1, case_id="case_123")
```

## 🧪 Testing

### Check Model Loading:
```bash
python manage.py shell
>>> from prediction import services
>>> models = services.load_prediction_models()
>>> print(models.keys())
# Should print: dict_keys(['preprocessor', 'outcome_model', 'lstm_predictor', 'vocab_size', 'max_length'])
```

### Test API:
```bash
# Health check
curl http://localhost:8000/api/prediction/health/

# Prediction (requires authentication)
curl -X POST http://localhost:8000/api/event-logs/1/prediction/all/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Token YOUR_TOKEN" \
  -d '{"source": "default", "case_id": "case_123"}'
```

## ⚠️ Important Notes

1. **NO Training**: This integration ONLY uses pre-trained models. Training must be done offline in Group 7's standalone environment.

2. **Model Path**: Models are in `prediction/trained_models/`. If you re-train models, replace these files.

3. **Dependencies**: Requires `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`, `pm4py`.

4. **Performance**: First prediction request takes longer (model loading). Subsequent requests are fast (cached).

5. **Memory**: Models stay loaded in memory after first use. This is intentional for performance.

## 🎨 UI Location

The Prediction step is accessible at:
- **URL**: `/prediction/`
- **Navigation**: Dashboard → Preprocessing → Discovery → Conformance → **Prediction**

## ✨ What We Did NOT Include

❌ Group 7's Django project structure  
❌ Group 7's templates (except adapted our own)  
❌ Group 7's `manage.py`, `urls.py`, etc.  
❌ Training code (`train_models.py`)  
❌ Testing code (`test_best_models.py`)  
❌ Their separate database or models  
❌ Their web pages (demo.html, etc.)  

## 📝 Code Comments

All service functions in `prediction/services.py` have comprehensive docstrings explaining:
- What the function does
- Where the Group 7 code is used
- How it integrates with main app
- Input parameters and return values
- Error handling

## ✅ Checklist

- [x] Created prediction app
- [x] Copied Group 7 core files (data_preprocessing, outcome_prediction, lstm_models)
- [x] Copied all trained models
- [x] Created service layer with lazy loading
- [x] Created DRF API endpoints
- [x] Created serializers for input validation
- [x] Created URL routing (API + web)
- [x] Registered app in settings.py
- [x] Included URLs in config/urls.py
- [x] Created prediction UI page
- [x] Integrated with existing EventLog infrastructure
- [x] Tested - no errors in code
- [x] Documentation complete

## 🎉 Integration Status: COMPLETE

The Prediction Engine (Group 7) is now fully integrated and ready to use!
