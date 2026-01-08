from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from preprocessing.urls import web_urlpatterns as preprocessing_web_urls
from discovery.urls import (
    web_urlpatterns as discovery_web_urls, 
    api_urlpatterns as discovery_api_urls, 
    job_urlpatterns as discovery_job_urls,
    csv_import_urlpatterns as csv_import_urls
)
from conformance.urls import web_urlpatterns as conformance_web_urls, api_urlpatterns as conformance_api_urls
from prediction.urls import web_urlpatterns as prediction_web_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('accounts.urls')),
    path('api/', include('accounts.api.urls')),  # REST API endpoints
    path('api/', include('preprocessing.urls')),  # Event Log & Preprocessing API
    path('api/', include(discovery_api_urls)),  # Discovery API
    path('api/', include(conformance_api_urls)),  # Conformance API (Group 6)
    path('api/', include('prediction.urls')),  # Prediction API (Group 7)    path('uploads/', include('uploads.urls')),  # File uploads for analysts
    path('translations/', include('translations.urls')),  # Language switcher
] + discovery_job_urls + csv_import_urls + preprocessing_web_urls + discovery_web_urls + conformance_web_urls + prediction_web_urls  # Add job URLs, CSV import URLs and web UIs

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
