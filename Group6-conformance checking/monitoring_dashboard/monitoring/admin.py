
from django.contrib import admin
from .models import ConformanceRun

@admin.register(ConformanceRun)
class ConformanceRunAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "model_file", "log_file", "compliant_percentage", "log_fitness", "precision")
    list_filter = ("model_file", "log_file")
    search_fields = ("model_file", "log_file")
