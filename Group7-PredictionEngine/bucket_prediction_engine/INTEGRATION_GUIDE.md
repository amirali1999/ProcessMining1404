# Integration Guide for Group 5 (Visualization)

## Overview

The **bucket_prediction_engine** provides REST API endpoints for real-time process predictions. This guide shows how to integrate with your visualization system.

## Quick Setup

### 1. Start the Prediction Server

```bash
# First, train the models (if not already done)
cd bucket_prediction_engine
python train_models.py ../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes

# Then start Django server (requires Django project setup)
# See "Django Integration" section below
```

### 2. Test API Availability

```bash
curl http://localhost:8000/api/health/
```

Expected response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "event_log_loaded": true
}
```

## API Endpoints

### 1. Predict Next Activity

**Endpoint**: `POST /api/predict/next-activity/`

**Request**:
```json
{
  "case_id": "H12345"
}
```

**Response**:
```json
{
  "case_id": "H12345",
  "current_prefix": ["Registration", "Triage", "XRay"],
  "predicted_next_activity": "Diagnosis",
  "top_k_candidates": [
    {"activity": "Diagnosis", "prob": 0.62},
    {"activity": "LabTest", "prob": 0.21},
    {"activity": "CTScan", "prob": 0.09},
    {"activity": "MRI", "prob": 0.05},
    {"activity": "Surgery", "prob": 0.03}
  ],
  "bucket_id": "3"
}
```

### 2. Predict Remaining Time

**Endpoint**: `POST /api/predict/remaining-time/`

**Request**:
```json
{
  "case_id": "H12345"
}
```

**Response**:
```json
{
  "case_id": "H12345",
  "current_prefix": ["Registration", "Triage", "XRay"],
  "predicted_remaining_time_hours": 12.5,
  "bucket_id": "3"
}
```

### 3. Predict Outcome

**Endpoint**: `POST /api/predict/outcome/`

**Request**:
```json
{
  "case_id": "H12345"
}
```

**Response**:
```json
{
  "case_id": "H12345",
  "current_prefix": ["Registration", "Triage", "XRay"],
  "predicted_outcome": "Discharged",
  "confidence": 0.85
}
```

### 4. Get All Predictions (Recommended)

**Endpoint**: `POST /api/predict/all/`

**Request**:
```json
{
  "case_id": "H12345"
}
```

**Response**:
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

## Alternative: Using Custom Prefix

If you don't have case_id, you can provide the activity sequence directly:

**Request**:
```json
{
  "prefix": ["Registration", "Triage", "XRay"]
}
```

This works for all endpoints.

## Visualization Ideas

### 1. Process Flow Highlighting

Use `top_k_candidates` to highlight probable next activities in the process model:

```javascript
// Example: Color-code activities by probability
const colorActivity = (activity, probability) => {
  if (probability > 0.5) return 'green';    // High probability
  if (probability > 0.2) return 'yellow';   // Medium probability
  return 'gray';                            // Low probability
};

response.predictions.next_activity.top_k_candidates.forEach(candidate => {
  highlightNode(candidate.activity, colorActivity(candidate.activity, candidate.prob));
});
```

### 2. Timeline Projection

Use `remaining_time` to show expected completion:

```javascript
const currentTime = new Date();
const remainingHours = response.predictions.remaining_time.hours;
const expectedCompletion = new Date(currentTime.getTime() + remainingHours * 60 * 60 * 1000);

displayTimeline({
  current: currentTime,
  expected: expectedCompletion,
  remaining: `${remainingHours.toFixed(1)} hours`
});
```

### 3. Outcome Badge

Use `outcome` prediction to show expected result:

```javascript
const outcome = response.predictions.outcome.prediction;
const confidence = response.predictions.outcome.confidence;

displayBadge({
  text: outcome,
  color: confidence > 0.8 ? 'success' : 'warning',
  label: `${(confidence * 100).toFixed(0)}% confidence`
});
```

### 4. Live Updates

Poll the API for active cases:

```javascript
const updatePredictions = async (caseId) => {
  const response = await fetch('/api/predict/all/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({case_id: caseId})
  });
  
  const data = await response.json();
  
  // Update UI
  updateProcessModel(data);
  updateTimeline(data);
  updateOutcomeBadge(data);
};

// Refresh every 30 seconds for active cases
setInterval(() => {
  activeCases.forEach(caseId => updatePredictions(caseId));
}, 30000);
```

## Django Integration

To use the API, you need to integrate it into a Django project:

### Step 1: Create Django Project

```bash
django-admin startproject prediction_api
cd prediction_api
```

### Step 2: Add to Django Settings

Edit `prediction_api/settings.py`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',  # Add this
]

# CORS (if frontend is on different domain)
INSTALLED_APPS += ['corsheaders']
MIDDLEWARE = ['corsheaders.middleware.CorsMiddleware'] + MIDDLEWARE
CORS_ALLOW_ALL_ORIGINS = True  # For development only!
```

### Step 3: Configure URLs

Edit `prediction_api/urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bucket_prediction_engine.urls')),  # Add this
]
```

### Step 4: Load Models on Startup

In `bucket_prediction_engine/api_views.py`, add initialization:

```python
from django.apps import AppConfig

class PredictionConfig(AppConfig):
    name = 'bucket_prediction_engine'
    
    def ready(self):
        from . import api_views
        api_views.load_models('trained_models')
        api_views.load_event_log('../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes')
```

### Step 5: Run Server

```bash
python manage.py runserver
```

## Example Client Code

### JavaScript (Fetch)

```javascript
async function predictNextActivity(caseId) {
  const response = await fetch('http://localhost:8000/api/predict/next-activity/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({case_id: caseId})
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
predictNextActivity('H12345').then(data => {
  console.log('Next activity:', data.predicted_next_activity);
  console.log('Top 3 candidates:', data.top_k_candidates.slice(0, 3));
});
```

### Python (Requests)

```python
import requests

def predict_all(case_id):
    response = requests.post(
        'http://localhost:8000/api/predict/all/',
        json={'case_id': case_id}
    )
    return response.json()

# Usage
predictions = predict_all('H12345')
print(f"Next activity: {predictions['predictions']['next_activity']['activity']}")
print(f"Remaining time: {predictions['predictions']['remaining_time']['hours']:.1f} hours")
print(f"Expected outcome: {predictions['predictions']['outcome']['prediction']}")
```

## Performance Considerations

1. **Caching**: Cache predictions for recently accessed cases
2. **Batch Processing**: If predicting for multiple cases, send requests in parallel
3. **Model Loading**: Models are loaded once at startup (not per request)
4. **Response Time**: Expect ~50-200ms per prediction depending on prefix length

## Error Handling

```javascript
async function safePrediction(caseId) {
  try {
    const response = await fetch('/api/predict/all/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({case_id: caseId})
    });
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction failed:', error);
    return {
      error: true,
      message: error.message,
      // Fallback data
      predictions: {
        next_activity: {activity: 'Unknown', top_k_candidates: []},
        remaining_time: {hours: null},
        outcome: {prediction: 'Unknown', confidence: 0}
      }
    };
  }
}
```

## Testing

Use curl to test all endpoints:

```bash
# Test health
curl http://localhost:8000/api/health/

# Test next activity
curl -X POST http://localhost:8000/api/predict/next-activity/ \
  -H "Content-Type: application/json" \
  -d '{"prefix": ["Registration", "Triage"]}'

# Test all predictions
curl -X POST http://localhost:8000/api/predict/all/ \
  -H "Content-Type: application/json" \
  -d '{"case_id": "A"}'
```

## Support

For issues or questions:
1. Check `bucket_prediction_engine/README.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for architecture details
3. See `COMPARISON.md` for differences between implementations

---

**Happy Integrating!** 🚀
