from django.urls import path
from .views import (
    RegisterAPIView,
    LoginAPIView,
    LogoutAPIView,
    MeAPIView,
    UsersListAPIView,
    RolesListAPIView,
)

urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='api_register'),
    path('login/', LoginAPIView.as_view(), name='api_login'),
    path('logout/', LogoutAPIView.as_view(), name='api_logout'),
    path('me/', MeAPIView.as_view(), name='api_me'),
    path('users/', UsersListAPIView.as_view(), name='api_users'),
    path('roles/', RolesListAPIView.as_view(), name='api_roles'),
]
