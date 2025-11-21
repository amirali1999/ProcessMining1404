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
