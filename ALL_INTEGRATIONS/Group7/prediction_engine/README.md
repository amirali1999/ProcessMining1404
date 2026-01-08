# Process Mining Prediction Engine

این پروژه یک موتور پیش‌بینی فرآیند (Process Mining Prediction Engine) است که با استفاده از یادگیری ماشین و شبکه‌های عصبی، نتایج فرآیندهای در حال اجرا را پیش‌بینی می‌کند.

## ویژگی‌ها

### فاز ۱: پیش‌بینی Outcome
- آموزش مدل‌های طبقه‌بندی (Decision Tree، Logistic Regression، Random Forest)
- استخراج ویژگی از پیشوندهای فرآیند (Case Prefixes)
- پیش‌بینی نتیجه نهایی یک Case بر اساس فعالیت‌های اولیه

### فاز ۲: پیش‌بینی فعالیت و زمان
- مدل LSTM برای پیش‌بینی فعالیت بعدی
- مدل LSTM برای پیش‌بینی زمان باقی‌مانده
- پیش‌بینی‌های توالی‌محور با دقت بالا

### API
- API RESTful با Django
- پیش‌بینی بر اساس case_id
- Endpoints برای تمام انواع پیش‌بینی

## نصب

### پیش‌نیازها
- Python 3.8 یا بالاتر
- pip

### نصب وابستگی‌ها

```bash
cd prediction_engine
pip install -r requirements.txt
```

## ساختار پروژه

```
prediction_engine/
├── data_preprocessing.py      # پیش‌پردازش داده XES
├── outcome_prediction.py      # مدل‌های پیش‌بینی Outcome (فاز 1)
├── lstm_models.py            # مدل‌های LSTM (فاز 2)
├── api_views.py              # Django REST API
├── train_models.py           # اسکریپت آموزش مدل‌ها
├── requirements.txt          # وابستگی‌های پروژه
└── README.md                 # این فایل
```

## استفاده

### 1. آموزش مدل‌ها

برای آموزش تمام مدل‌ها با داده XES:

```bash
python train_models.py path/to/your/event_log.xes --output-dir trained_models
```

فقط مدل‌های Outcome:
```bash
python train_models.py path/to/your/event_log.xes --outcome-only
```

فقط مدل‌های LSTM:
```bash
python train_models.py path/to/your/event_log.xes --lstm-only
```

#### مثال با Hospital Billing Dataset:

```bash
python train_models.py "../Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes"
```

### 2. استفاده از API

#### راه‌اندازی سرور Django

ابتدا باید یک پروژه Django ایجاد کنید و API را پیکربندی کنید:

```bash
# ایجاد پروژه Django
django-admin startproject prediction_project
cd prediction_project

# کپی فایل‌های API
cp ../prediction_engine/api_views.py .

# در settings.py اضافه کنید:
INSTALLED_APPS += ['rest_framework']

# در urls.py:
from api_views import *
from django.urls import path

urlpatterns = [
    path('api/predict/outcome/', predict_outcome),
    path('api/predict/next-activity/', predict_next_activity),
    path('api/predict/remaining-time/', predict_remaining_time),
    path('api/predict/all/', predict_all),
    path('api/health/', health_check),
]

# بارگذاری مدل‌ها هنگام راه‌اندازی
from api_views import load_models
load_models()

# اجرای سرور
python manage.py runserver
```

#### Endpoints

**1. پیش‌بینی Outcome**
```bash
POST http://localhost:8000/api/predict/outcome/
Content-Type: application/json

{
    "case_id": "case_123"
}
```

پاسخ:
```json
{
    "case_id": "case_123",
    "predicted_outcome": "APPROVED",
    "current_activities": ["Register", "Check", "Verify"],
    "prefix_length": 3
}
```

**2. پیش‌بینی فعالیت بعدی**
```bash
POST http://localhost:8000/api/predict/next-activity/
Content-Type: application/json

{
    "case_id": "case_123"
}
```

پاسخ:
```json
{
    "case_id": "case_123",
    "predicted_next_activity": "Approve",
    "current_activities": ["Register", "Check", "Verify"],
    "top_predictions": [
        {"activity": "Approve", "probability": 0.75},
        {"activity": "Reject", "probability": 0.15},
        {"activity": "Review", "probability": 0.10}
    ]
}
```

**3. پیش‌بینی زمان باقی‌مانده**
```bash
POST http://localhost:8000/api/predict/remaining-time/
Content-Type: application/json

{
    "case_id": "case_123"
}
```

پاسخ:
```json
{
    "case_id": "case_123",
    "predicted_remaining_time_seconds": 86400,
    "predicted_remaining_time_hours": 24,
    "predicted_remaining_time_days": 1,
    "elapsed_time_seconds": 43200,
    "current_activities": ["Register", "Check", "Verify"]
}
```

**4. تمام پیش‌بینی‌ها**
```bash
POST http://localhost:8000/api/predict/all/
Content-Type: application/json

{
    "case_id": "case_123"
}
```

### 3. استفاده به صورت Standalone

می‌توانید مدل‌ها را مستقیماً در کد Python خود استفاده کنید:

```python
from data_preprocessing import XESDataPreprocessor
from outcome_prediction import EnsembleOutcomePredictor
from lstm_models import CombinedLSTMPredictor

# بارگذاری مدل‌ها
ensemble = EnsembleOutcomePredictor()
ensemble.load('trained_models/ensemble')

lstm_predictor = CombinedLSTMPredictor(vocab_size=50, max_length=50)
lstm_predictor.load('trained_models/lstm')

# بارگذاری preprocessor
preprocessor = XESDataPreprocessor('')
preprocessor.load_preprocessor('trained_models/preprocessor.pkl')

# استفاده
# ... آماده‌سازی داده ...
outcome = ensemble.predict(features)
next_activity = lstm_predictor.next_activity_model.predict(sequence)
remaining_time = lstm_predictor.remaining_time_model.predict(sequence)
```

## توضیحات تکنیکی

### پیش‌پردازش داده
- خواندن فایل XES با pm4py
- پاک‌سازی داده (حذف تکرار، مدیریت مقادیر گمشده)
- ایجاد prefixes با طول‌های مختلف
- استخراج ویژگی‌های آماری و توالی‌ای

### مدل‌های فاز ۱
- **Decision Tree**: مدل درختی با عمق محدود
- **Logistic Regression**: رگرسیون لجستیک چند کلاسه
- **Random Forest**: جنگل تصادفی با 100 درخت
- **Ensemble**: ترکیب نتایج با رأی‌گیری

### مدل‌های فاز ۲
- **Next Activity LSTM**:
  - Embedding layer برای فعالیت‌ها
  - 2 لایه LSTM با dropout
  - خروجی softmax برای طبقه‌بندی
  
- **Remaining Time LSTM**:
  - معماری مشابه
  - خروجی linear برای رگرسیون
  - پیش‌بینی زمان به ثانیه

### ارزیابی
- **Outcome**: Accuracy, F1-Score, Confusion Matrix
- **Next Activity**: Accuracy, Top-K Accuracy
- **Remaining Time**: MAE, RMSE, MAPE

## Dataset

این کد برای Hospital Billing Event Log طراحی شده است:
- 100,000 traces
- چندین attribute (caseType, diagnosis, closeCode, etc.)
- فرآیند billing خدمات پزشکی

می‌توانید با هر XES event log دیگری نیز استفاده کنید.

## نکات مهم

1. **حجم داده**: برای dataset‌های بزرگ، آموزش ممکن است ساعت‌ها طول بکشد
2. **حافظه**: LSTM مدل‌ها به GPU دسترسی داشته باشند بهتر است
3. **Hyperparameters**: می‌توانید در کد مدل‌ها تنظیم کنید
4. **Preprocessing**: برای dataset‌های جدید ممکن است نیاز به تنظیم باشد

## تکنولوژی‌ها

- **Python 3.8+**
- **pm4py**: خواندن و پردازش XES
- **scikit-learn**: مدل‌های یادگیری ماشین
- **TensorFlow/Keras**: شبکه‌های عصبی LSTM
- **pandas/numpy**: پردازش داده
- **Django REST Framework**: API

## مشارکت

برای بهبود پروژه:
1. Feature engineering بیشتر
2. مدل‌های پیشرفته‌تر (Transformer, GNN)
3. Hyperparameter tuning
4. Cross-validation
5. Feature importance analysis

## مجوز

این پروژه برای اهداف آموزشی در دانشگاه ایجاد شده است.

## تماس

برای سوالات و پشتیبانی، با تیم توسعه تماس بگیرید.

---

**گروه ۷: موتور پیش‌بینی فرآیند**
**دانشگاه - درس Process Mining**
**سال 2025**
