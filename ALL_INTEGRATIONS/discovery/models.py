from django.db import models
from django.conf import settings
from uploads.models import EventLog


class CSVImportSession(models.Model):
    """
    Temporary session for CSV import with column mapping.
    Stores uploaded CSV file before final processing.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='csv_import_sessions',
        help_text="User who uploaded this CSV"
    )
    
    uploaded_file = models.FileField(
        upload_to='temp_csv_imports/%Y/%m/%d/',
        help_text="Temporarily stored CSV file"
    )
    
    original_filename = models.CharField(
        max_length=255,
        help_text="Original CSV filename"
    )
    
    project_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Project name for this import"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this session was created"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'CSV Import Session'
        verbose_name_plural = 'CSV Import Sessions'
    
    def __str__(self):
        return f"CSV Import {self.id}: {self.original_filename}"


class DiscoveredProcessModel(models.Model):
    """
    Stores a discovered process model (Petri net) in PNML format.
    Used by Groups 5 (Visualization) and 6 (Conformance Checking).
    """
    ALGORITHM_CHOICES = [
        ('alpha', 'Alpha Miner'),
        ('heuristics', 'Heuristics Miner'),
        ('inductive', 'Inductive Miner'),  # For future use
    ]
    
    SOURCE_CHOICES = [
        ('raw', 'Raw Data'),
        ('cleaned', 'Cleaned Data'),
    ]
    
    # Relationships
    event_log = models.ForeignKey(
        EventLog,
        on_delete=models.CASCADE,
        related_name='discovered_models',
        help_text="The event log this model was discovered from"
    )
    
    discovered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='discovered_models',
        help_text="User who ran the discovery algorithm"
    )
    
    # Discovery parameters
    algorithm = models.CharField(
        max_length=20,
        choices=ALGORITHM_CHOICES,
        help_text="The discovery algorithm used"
    )
    
    source_version = models.CharField(
        max_length=10,
        choices=SOURCE_CHOICES,
        default='raw',
        help_text="Whether raw or cleaned log was used"
    )
    
    # Model data (PNML XML format)
    pnml_content = models.TextField(
        help_text="The Petri net in PNML XML format (for Groups 5 & 6)"
    )
    
    # Model statistics (cached for quick access)
    num_places = models.IntegerField(
        default=0,
        help_text="Number of places in the Petri net"
    )
    
    num_transitions = models.IntegerField(
        default=0,
        help_text="Number of transitions in the Petri net"
    )
    
    num_arcs = models.IntegerField(
        default=0,
        help_text="Number of arcs in the Petri net"
    )
    
    # Metadata
    discovered_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this model was discovered"
    )
    
    class Meta:
        ordering = ['-discovered_at']
        verbose_name = 'Discovered Process Model'
        verbose_name_plural = 'Discovered Process Models'
        indexes = [
            models.Index(fields=['event_log', 'algorithm']),        models.Index(fields=['event_log', 'source_version']),
        ]
    
    def __str__(self):
        return f"{self.get_algorithm_display()} on {self.event_log.name} ({self.source_version})"
    
    @property
    def complexity_score(self):
        """Simple complexity metric for the model"""
        return self.num_places + self.num_transitions + self.num_arcs


class EventLogJob(models.Model):
    """
    Represents a background job for uploading, cleaning, and mining an event log.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('done', 'Done'),
        ('error', 'Error'),
    ]
    
    # Mining method choices (scalable - add more methods here)
    MINING_METHOD_CHOICES = [
        ('alpha', 'Alpha Miner'),
        ('heuristics', 'Heuristics Miner'),
    ]
    
    # Relationships
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='event_log_jobs',
        help_text="User who uploaded this file"
    )
      # File information
    original_file = models.FileField(
        upload_to='uploads/event_logs/%Y/%m/%d/',
        help_text="The uploaded event log file (.xes or .csv)"
    )
    original_filename = models.CharField(
        max_length=255,
        help_text="Original name of the uploaded file"
    )
    
    # Project name (user-defined)
    project_name = models.CharField(
        max_length=255,
        blank=True,
        default='',
        help_text="User-defined project name"
    )
    
    # Processing options
    cleaning_enabled = models.BooleanField(
        default=False,
        help_text="Whether data cleaning was requested"
    )
    mining_method = models.CharField(
        max_length=20,
        choices=MINING_METHOD_CHOICES,
        help_text="The process mining method to use"
    )
    
    # Job status
    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='pending',
        help_text="Current job status"
    )
    progress = models.IntegerField(
        default=0,
        help_text="Progress percentage (0-100)"
    )
    message = models.CharField(
        max_length=255,
        blank=True,
        help_text="Current status message"
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error details if job failed"
    )
    
    # Output
    output_map_image = models.ImageField(
        upload_to='outputs/process_maps/%Y/%m/%d/',
        null=True,
        blank=True,
        help_text="Generated process map as PNG"
    )
    output_map_svg = models.FileField(
        upload_to='outputs/process_maps/%Y/%m/%d/',
        null=True,
        blank=True,
        help_text="Generated process map as SVG"
    )
    
    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this job was created"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last status update time"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Event Log Job'
        verbose_name_plural = 'Event Log Jobs'
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Job {self.id}: {self.original_filename} ({self.status})"
    
    def get_output_url(self):
        """Returns the URL of the generated process map"""
        if self.output_map_svg:
            return self.output_map_svg.url
        elif self.output_map_image:
            return self.output_map_image.url
        return None
