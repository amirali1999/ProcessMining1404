from django.db import models

class EventLog(models.Model):
    log_name = models.CharField(max_length=100, blank=True, null=True)
    case_id = models.CharField(max_length=100)
    activity = models.CharField(max_length=200)
    timestamp = models.DateTimeField()
    
    def __str__(self):
        return f"{self.case_id} - {self.activity}"
