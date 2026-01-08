"""
Django URLs Configuration for Prediction API
"""

from django.urls import path
from . import api_views

# Initialize models on startup
api_views.load_models()

urlpatterns = [
    path('', api_views.demo_page, name='demo_page'),
    path('api/predict/outcome/', api_views.predict_outcome, name='predict_outcome'),
    path('api/predict/next-activity/', api_views.predict_next_activity, name='predict_next_activity'),
    path('api/predict/remaining-time/', api_views.predict_remaining_time, name='predict_remaining_time'),
    path('api/predict/all/', api_views.predict_all, name='predict_all'),
    path('api/health/', api_views.health_check, name='health_check'),
]
