"""
Conformance Checking Models (Group 6)

Stores token replay conformance results for process compliance analysis.
"""
from django.db import models
from uploads.models import EventLog
from discovery.models import DiscoveredProcessModel


class ConformanceResult(models.Model):
    """
    Stores the result of a token replay conformance check.
    
    Used by Group 6 to analyze which cases comply with the discovered process model.
    """
    SOURCE_CHOICES = [
        ('raw', 'Raw Data'),
        ('cleaned', 'Cleaned Data'),
        ('default', 'Default'),
    ]
    
    # Relationships
    event_log = models.ForeignKey(
        EventLog,
        on_delete=models.CASCADE,
        related_name='conformance_results',
        help_text="The event log used for conformance checking"
    )
    
    discovered_model = models.ForeignKey(
        DiscoveredProcessModel,
        on_delete=models.CASCADE,
        related_name='conformance_results',
        help_text="The process model checked against"
    )
    
    source_version = models.CharField(
        max_length=16,
        choices=SOURCE_CHOICES,
        default='default',
        help_text="Which version of the log was used"
    )
    
    # Metadata
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this conformance check was performed"
    )
    
    # Statistics
    total_cases = models.IntegerField(
        help_text="Total number of cases analyzed"
    )
    
    compliant_cases = models.IntegerField(
        help_text="Number of cases that fit the model"
    )
    
    non_compliant_cases = models.IntegerField(
        help_text="Number of cases that don't fit the model"
    )
    
    compliant_percentage = models.FloatField(
        help_text="Percentage of compliant cases"
    )
    
    non_compliant_percentage = models.FloatField(
        help_text="Percentage of non-compliant cases"
    )
    
    # Case IDs (stored as JSON for SQLite compatibility)
    compliant_case_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of compliant case IDs"
    )
    
    non_compliant_case_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of non-compliant case IDs"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Conformance Result'
        verbose_name_plural = 'Conformance Results'
        indexes = [
            models.Index(fields=['event_log', 'discovered_model']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Conformance: {self.event_log.name} vs {self.discovered_model} ({self.compliant_percentage:.1f}% fit)"
