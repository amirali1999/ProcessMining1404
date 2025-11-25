from django.db import models
from django.contrib.auth.models import User
from storages.backends.s3boto3 import S3Boto3Storage


class EventLog(models.Model):

    file = models.FileField(
        upload_to='raw_logs/',
        storage=S3Boto3Storage(),
        help_text='فایل Event Log (CSV, XES)'
    )
    original_filename = models.CharField(
        max_length=255,
        help_text='نام اصلی فایل آپلود شده'
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text='زمان آپلود'
    )
    file_size = models.BigIntegerField(
        null=True,
        blank=True,
        help_text='حجم فایل به بایت'
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='uploaded_logs',
        help_text='کاربر آپلودکننده'
    )
    
    class Meta:
        db_table = 'event_logs'
        verbose_name = 'Event Log'
        verbose_name_plural = 'Event Logs'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.original_filename} ({self.uploaded_at.strftime('%Y-%m-%d %H:%M')})"
    
    def get_file_size_mb(self):
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
