from rest_framework import serializers
from .models import EventLog

class EventLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = EventLog
        fields = ['id', 'log_name', 'case_id', 'activity', 'timestamp']
