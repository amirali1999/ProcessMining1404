from django.urls import path
from . import views

urlpatterns = [
    path('', views.petri_net_view, name='petri-net'),  # root of website
    
    path('dashboard', views.dashboard_view, name='dashboard'),
   
    path('upload', views.upload_view, name='upload'),
]
