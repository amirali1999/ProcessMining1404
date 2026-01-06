
from django.db import models

class ConformanceRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    model_file = models.CharField(max_length=255)
    log_file = models.CharField(max_length=255)
    timestamp_key_used = models.CharField(max_length=255, blank=True, default="")

    total_cases = models.IntegerField()
    compliant_cases = models.IntegerField()
    non_compliant_cases = models.IntegerField()
    compliant_percentage = models.FloatField()
    non_compliant_percentage = models.FloatField()

    log_fitness = models.FloatField(null=True, blank=True)
    average_trace_fitness = models.FloatField(null=True, blank=True)
    percentage_of_fitting_traces = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)

    summary_json_path = models.CharField(max_length=500, blank=True, default="")
    compliant_csv_path = models.CharField(max_length=500, blank=True, default="")
    non_compliant_csv_path = models.CharField(max_length=500, blank=True, default="")
    pnml_path = models.CharField(max_length=500, blank=True, default="")
    log_csv_path = models.CharField(max_length=500, blank=True, default="")

    def __str__(self):
        return f"{self.model_file} | {self.log_file} | {self.created_at:%Y-%m-%d %H:%M}"
