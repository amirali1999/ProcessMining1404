from django.apps import AppConfig


class ConformanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'conformance'
    verbose_name = ''  # Hide app grouping in admin
