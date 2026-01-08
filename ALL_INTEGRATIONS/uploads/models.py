from django.db import models
from django.conf import settings


class UploadedFile(models.Model):
    uploader = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to='uploads/%Y/%m/%d')
    original_name = models.CharField(max_length=255)
    description = models.CharField(max_length=255, blank=True)
    content_type = models.CharField(max_length=100, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self) -> str:
        return self.original_name

    @property
    def size_kb(self) -> float:
        try:
            return round(self.file.size / 1024, 2)
        except Exception:
            return 0.0

    @property
    def size_mb(self) -> float:
        """Return file size in megabytes"""
        try:
            return round(self.size_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0

    @property
    def extension(self) -> str:
        try:
            name = self.original_name
            return (name.rsplit('.', 1)[-1] if '.' in name else '').lower()
        except Exception:
            return ''


class EventLog(models.Model):
    """
    Represents a Process Mining Event Log.
    This is the canonical model that all groups (2-7) will reference.
    """
    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('xes', 'XES'),
        ('parquet', 'Parquet'),
    ]
    
    SOURCE_CHOICES = [
        ('raw', 'Raw Data'),
        ('cleaned', 'Cleaned Data'),
    ]
    
    # Basic info
    name = models.CharField(max_length=255, help_text="Human-readable log name")
    uploaded_file = models.OneToOneField(
        UploadedFile, 
        on_delete=models.CASCADE, 
        related_name='event_log',
        help_text="Reference to the uploaded file (Group 2)"
    )
    
    # File metadata
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES, default='csv')
    
    # Cleaned version
    cleaned_file_path = models.FileField(
        upload_to='cleaned_logs/%Y/%m/%d',
        blank=True,
        null=True,
        help_text="Path to cleaned/preprocessed log file (created by Group 3)"
    )
    
    # Metadata (JSON field for flexibility)
    meta_info = models.JSONField(
        default=dict,
        blank=True,
        help_text="Metadata: cases, events, activities, time_range, variants, etc."
    )
    
    # Default source for downstream groups (4-7)
    default_source_for_downstream = models.CharField(
        max_length=10,
        choices=SOURCE_CHOICES,
        default='raw',
        help_text="Which version to use for discovery/conformance/prediction"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Event Log'
        verbose_name_plural = 'Event Logs'
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def original_file_path(self) -> str:
        """Convenience accessor for the original uploaded file path"""
        return self.uploaded_file.file.path if self.uploaded_file else ''
    
    @property
    def has_cleaned_version(self) -> bool:
        """Check if a cleaned version exists"""
        return bool(self.cleaned_file_path)
    
    @property
    def num_cases(self) -> int:
        """Number of cases from metadata"""
        return self.meta_info.get('num_cases', 0)
    
    @property
    def num_events(self) -> int:
        """Number of events from metadata"""
        return self.meta_info.get('num_events', 0)
    
    @property
    def num_activities(self) -> int:
        """Number of unique activities from metadata"""
        return self.meta_info.get('num_activities', 0)
