from django.contrib import admin
from uploads.models import EventLog


@admin.register(EventLog)
class EventLogAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_type', 'has_cleaned_version', 'default_source_for_downstream', 'created_at']
    list_filter = ['file_type', 'default_source_for_downstream', 'created_at']
    search_fields = ['name', 'uploaded_file__original_name']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = [
        ('Basic Info', {
            'fields': ['name', 'uploaded_file', 'file_type']
        }),
        ('Preprocessing', {
            'fields': ['cleaned_file_path', 'default_source_for_downstream']
        }),
        ('Metadata', {
            'fields': ['meta_info'],
            'classes': ['collapse']
        }),
        ('Timestamps', {
            'fields': ['created_at', 'updated_at'],
            'classes': ['collapse']
        }),
    ]
