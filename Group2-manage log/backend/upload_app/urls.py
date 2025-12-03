from django.urls import path
from . import views

app_name = 'upload_app'

urlpatterns = [
    path('', views.upload_event_log, name='upload'),
    path('success/<int:pk>/', views.upload_success, name='upload_success'),
    path('list/', views.event_log_list, name='list'),
    path('detail/<int:pk>/', views.event_log_detail, name='detail'),
]
