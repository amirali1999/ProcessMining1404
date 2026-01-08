from django.urls import path
from . import views

urlpatterns = [
    path('set-language/', views.set_language, name='set_language'),
]
