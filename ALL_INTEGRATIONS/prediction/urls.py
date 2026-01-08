"""
URL Configuration for Prediction API and Web Views
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'prediction'

# API URL patterns (for /api/ includes)
api_urlpatterns = [
    # Prediction endpoints
    path('event-logs/<int:pk>/prediction/all/', views.PredictionViewSet.as_view({'post': 'predict_all'}), name='predict-all'),
    path('event-logs/<int:pk>/prediction/outcome/', views.PredictionViewSet.as_view({'post': 'predict_outcome'}), name='predict-outcome'),
    path('event-logs/<int:pk>/prediction/next-activity/', views.PredictionViewSet.as_view({'post': 'predict_next_activity'}), name='predict-next-activity'),
    path('event-logs/<int:pk>/prediction/remaining-time/', views.PredictionViewSet.as_view({'post': 'predict_remaining_time'}), name='predict-remaining-time'),
    path('event-logs/<int:pk>/prediction/cases/', views.get_cases_view, name='get-cases'),
    path('prediction/health/', views.PredictionViewSet.as_view({'get': 'health_check'}), name='health'),
]

# Web URL patterns (for root includes)
web_urlpatterns = [
    path('prediction/', views.prediction_page, name='prediction'),
]

# Default to API patterns
urlpatterns = api_urlpatterns
